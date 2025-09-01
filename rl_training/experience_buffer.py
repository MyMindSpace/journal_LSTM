# rl_training/experience_buffer.py
# core/context_assembler.py
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import random
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from collections import deque
import heapq
import json
import asyncio
from datetime import datetime, timedelta
import logging

from models.rl_experience import RLExperience, ExperienceBatch
from database.astra_connector import AstraDBConnector

logger = logging.getLogger(__name__)

class ExperienceBuffer:
    """
    Base experience buffer for storing and sampling RL experiences
    """
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.position = 0
        
    def add(self, experience: RLExperience):
        """Add experience to buffer"""
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> ExperienceBatch:
        """Sample random batch of experiences"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        experiences = random.sample(list(self.buffer), batch_size)
        return ExperienceBatch(experiences)
    
    def size(self) -> int:
        """Get current buffer size"""
        return len(self.buffer)
    
    def is_ready(self, min_size: int = 100) -> bool:
        """Check if buffer has enough experiences for training"""
        return len(self.buffer) >= min_size
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
        self.position = 0


class PrioritizedExperienceBuffer:
    """
    Prioritized Experience Replay buffer
    Samples experiences based on TD error magnitude
    """
    
    def __init__(self, max_size: int = 10000, alpha: float = 0.6, beta_start: float = 0.4, beta_end: float = 1.0):
        self.max_size = max_size
        self.alpha = alpha  # Priority exponent
        self.beta_start = beta_start  # Importance sampling start
        self.beta_end = beta_end  # Importance sampling end
        self.beta_schedule_steps = 100000  # Steps to anneal beta
        
        # Storage
        self.buffer = []
        self.priorities = []
        self.position = 0
        self.max_priority = 1.0
        
        # Statistics
        self.sample_count = 0
        
    def add(self, experience: RLExperience, td_error: Optional[float] = None):
        """Add experience with priority"""
        # Calculate priority from TD error or use max priority for new experiences
        if td_error is not None:
            priority = (abs(td_error) + 1e-6) ** self.alpha
        else:
            priority = self.max_priority
        
        experience.priority = priority
        experience.td_error = td_error
        
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority
        
        # Update max priority
        self.max_priority = max(self.max_priority, priority)
        self.position = (self.position + 1) % self.max_size
    
    def sample(self, batch_size: int) -> Tuple[ExperienceBatch, List[int], List[float]]:
        """Sample batch with importance sampling weights"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities[:len(self.buffer)])
        probabilities = priorities / np.sum(priorities)
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)
        
        # Calculate importance sampling weights
        beta = self._get_beta()
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights = weights / np.max(weights)  # Normalize weights
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        batch = ExperienceBatch(experiences, weights.tolist())
        
        self.sample_count += 1
        return batch, indices.tolist(), weights.tolist()
    
    def update_priorities(self, indices: List[int], td_errors: List[float]):
        """Update priorities based on new TD errors"""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.buffer[idx].priority = priority
            self.buffer[idx].td_error = td_error
            
            # Update max priority
            self.max_priority = max(self.max_priority, priority)
    
    def _get_beta(self) -> float:
        """Get current beta value (annealed from beta_start to beta_end)"""
        progress = min(1.0, self.sample_count / self.beta_schedule_steps)
        return self.beta_start + progress * (self.beta_end - self.beta_start)
    
    def size(self) -> int:
        """Get current buffer size"""
        return len(self.buffer)
    
    def is_ready(self, min_size: int = 100) -> bool:
        """Check if buffer has enough experiences for training"""
        return len(self.buffer) >= min_size
    
    def get_priority_stats(self) -> Dict[str, float]:
        """Get priority statistics"""
        if not self.priorities:
            return {}
        
        priorities = np.array(self.priorities[:len(self.buffer)])
        return {
            'mean_priority': np.mean(priorities),
            'max_priority': np.max(priorities),
            'min_priority': np.min(priorities),
            'std_priority': np.std(priorities),
            'current_beta': self._get_beta()
        }


class PersistentExperienceBuffer:
    """
    Experience buffer that persists to Astra DB
    Combines in-memory buffer with database storage
    """
    
    def __init__(self, 
                 db_connector: AstraDBConnector,
                 memory_buffer_size: int = 1000,
                 max_db_experiences: int = 100000,
                 cleanup_interval_hours: int = 24):
        
        self.db_connector = db_connector
        self.memory_buffer_size = memory_buffer_size
        self.max_db_experiences = max_db_experiences
        self.cleanup_interval = timedelta(hours=cleanup_interval_hours)
        
        # In-memory buffer for fast access
        self.memory_buffer = PrioritizedExperienceBuffer(memory_buffer_size)
        
        # Track when we last cleaned up old experiences
        self.last_cleanup = datetime.now()
        
        # Statistics
        self.stats = {
            'experiences_added': 0,
            'experiences_loaded': 0,
            'database_saves': 0,
            'cleanup_runs': 0
        }
    
    async def add(self, experience: RLExperience, td_error: Optional[float] = None):
        """Add experience to both memory buffer and database"""
        # Add to memory buffer
        self.memory_buffer.add(experience, td_error)
        
        # Save to database asynchronously
        try:
            await self._save_experience_to_db(experience)
            self.stats['database_saves'] += 1
        except Exception as e:
            logger.error(f"Failed to save experience to database: {e}")
        
        self.stats['experiences_added'] += 1
        
        # Periodic cleanup
        if datetime.now() - self.last_cleanup > self.cleanup_interval:
            asyncio.create_task(self._cleanup_old_experiences())
    
    async def sample(self, batch_size: int, prefer_memory: bool = True) -> Tuple[ExperienceBatch, List[int], List[float]]:
        """Sample experiences, preferring memory buffer for speed"""
        
        if prefer_memory and self.memory_buffer.is_ready(batch_size):
            # Sample from memory buffer
            return self.memory_buffer.sample(batch_size)
        else:
            # Load experiences from database if needed
            await self._load_experiences_from_db(batch_size * 2)  # Load extra for diversity
            return self.memory_buffer.sample(batch_size)
    
    async def load_user_experiences(self, user_id: str, limit: int = 500) -> List[RLExperience]:
        """Load recent experiences for a specific user"""
        query = """
        SELECT * FROM rl_experiences 
        WHERE user_id = ? 
        ORDER BY created_at DESC 
        LIMIT ?
        """
        
        try:
            rows = await self.db_connector.execute_async(query, (user_id, limit))
            experiences = []
            
            for row in rows:
                experience = RLExperience.from_dict(dict(row))
                experiences.append(experience)
            
            self.stats['experiences_loaded'] += len(experiences)
            return experiences
            
        except Exception as e:
            logger.error(f"Error loading user experiences: {e}")
            return []
    
    async def get_buffer_stats(self) -> Dict[str, Any]:
        """Get comprehensive buffer statistics"""
        memory_stats = self.memory_buffer.get_priority_stats()
        
        # Get database stats
        db_stats = await self._get_database_stats()
        
        return {
            'memory_buffer': {
                'size': self.memory_buffer.size(),
                'max_size': self.memory_buffer.max_size,
                'priority_stats': memory_stats
            },
            'database': db_stats,
            'overall_stats': self.stats,
            'last_cleanup': self.last_cleanup.isoformat()
        }
    
    async def _save_experience_to_db(self, experience: RLExperience):
        """Save single experience to database"""
        experience_dict = experience.to_dict()
        
        insert_query = """
        INSERT INTO rl_experiences (
            id, user_id, episode_id, step_id, state, action, reward, 
            next_state, done, priority, td_error, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        await self.db_connector.execute_async(insert_query, (
            experience.id,
            experience.user_id,
            experience.episode_id,
            experience.step_id,
            json.dumps(experience_dict['state']),
            json.dumps(experience_dict['action']),
            json.dumps(experience_dict['reward']),
            json.dumps(experience_dict['next_state']) if experience_dict['next_state'] else None,
            experience.done,
            experience.priority,
            experience.td_error,
            experience.created_at
        ))
    
    async def _load_experiences_from_db(self, limit: int = 1000):
        """Load recent high-priority experiences from database to memory buffer"""
        query = """
        SELECT * FROM rl_experiences 
        ORDER BY priority DESC, created_at DESC 
        LIMIT ?
        """
        
        try:
            rows = await self.db_connector.execute_async(query, (limit,))
            
            for row in rows:
                row_dict = dict(row)
                
                # Parse JSON fields
                row_dict['state'] = json.loads(row_dict['state'])
                row_dict['action'] = json.loads(row_dict['action'])
                row_dict['reward'] = json.loads(row_dict['reward'])
                if row_dict['next_state']:
                    row_dict['next_state'] = json.loads(row_dict['next_state'])
                
                experience = RLExperience.from_dict(row_dict)
                
                # Add to memory buffer if not already there
                if not any(exp.id == experience.id for exp in self.memory_buffer.buffer):
                    self.memory_buffer.add(experience, experience.td_error)
            
            self.stats['experiences_loaded'] += len(rows)
            
        except Exception as e:
            logger.error(f"Error loading experiences from database: {e}")
    
    async def _cleanup_old_experiences(self):
        """Remove old experiences from database to maintain size limits"""
        try:
            # Count current experiences
            count_query = "SELECT COUNT(*) as count FROM rl_experiences"
            count_result = await self.db_connector.execute_async(count_query)
            current_count = count_result[0]['count'] if count_result else 0
            
            if current_count > self.max_db_experiences:
                # Delete oldest experiences
                experiences_to_delete = current_count - self.max_db_experiences
                
                delete_query = """
                DELETE FROM rl_experiences 
                WHERE id IN (
                    SELECT id FROM rl_experiences 
                    ORDER BY created_at ASC 
                    LIMIT ?
                )
                """
                
                await self.db_connector.execute_async(delete_query, (experiences_to_delete,))
                
                logger.info(f"Cleaned up {experiences_to_delete} old experiences from database")
            
            self.last_cleanup = datetime.now()
            self.stats['cleanup_runs'] += 1
            
        except Exception as e:
            logger.error(f"Error during experience cleanup: {e}")
    
    async def _get_database_stats(self) -> Dict[str, Any]:
        """Get statistics from database"""
        try:
            stats_query = """
            SELECT 
                COUNT(*) as total_experiences,
                COUNT(DISTINCT user_id) as unique_users,
                COUNT(DISTINCT episode_id) as unique_episodes,
                AVG(priority) as avg_priority,
                MAX(created_at) as latest_experience,
                MIN(created_at) as oldest_experience
            FROM rl_experiences
            """
            
            result = await self.db_connector.execute_async(stats_query)
            
            if result:
                return dict(result[0])
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}


class EpisodeBuffer:
    """
    Buffer for managing complete episodes (conversation sessions)
    Useful for episode-based learning algorithms
    """
    
    def __init__(self, max_episodes: int = 1000):
        self.max_episodes = max_episodes
        self.episodes: Dict[str, List[RLExperience]] = {}
        self.completed_episodes = deque(maxlen=max_episodes)
        
    def add_experience(self, experience: RLExperience):
        """Add experience to current episode"""
        episode_id = experience.episode_id
        
        if episode_id not in self.episodes:
            self.episodes[episode_id] = []
        
        self.episodes[episode_id].append(experience)
        
        # If episode is done, move to completed episodes
        if experience.done:
            self.completed_episodes.append(self.episodes[episode_id])
            del self.episodes[episode_id]
    
    def get_completed_episodes(self, num_episodes: int) -> List[List[RLExperience]]:
        """Get recent completed episodes"""
        if len(self.completed_episodes) < num_episodes:
            num_episodes = len(self.completed_episodes)
        
        return list(self.completed_episodes)[-num_episodes:]
    
    def get_episode_batch(self, batch_size: int) -> ExperienceBatch:
        """Sample experiences from completed episodes"""
        all_experiences = []
        
        for episode in self.completed_episodes:
            all_experiences.extend(episode)
        
        if len(all_experiences) < batch_size:
            batch_size = len(all_experiences)
        
        experiences = random.sample(all_experiences, batch_size)
        return ExperienceBatch(experiences)
    
    def get_stats(self) -> Dict[str, int]:
        """Get episode buffer statistics"""
        return {
            'active_episodes': len(self.episodes),
            'completed_episodes': len(self.completed_episodes),
            'total_experiences': sum(len(ep) for ep in self.completed_episodes),
            'avg_episode_length': np.mean([len(ep) for ep in self.completed_episodes]) if self.completed_episodes else 0
        }