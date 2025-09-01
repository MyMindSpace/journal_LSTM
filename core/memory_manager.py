# core/memory_manager.py
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import asyncio
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import uuid
import logging
import numpy as np

from models.memory_item import MemoryItem
from database.astra_connector import AstraDBConnector
from database.memory_store import MemoryStore
from core.gate_networks import LSTMGateNetwork
from core.context_assembler import ContextAssembler
from config.settings import LSTMConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Main orchestrator for LSTM memory management
    Handles memory creation, updates, retrieval, and decay
    """
    
    def __init__(self, config: LSTMConfig):
        self.config = config
        
        # Initialize components
        self.gate_network = LSTMGateNetwork(
            input_size=config.gate_network.input_size,
            hidden_size=config.gate_network.hidden_size,
            dropout=config.gate_network.dropout_rate
        )
        
        self.db_connector = AstraDBConnector(config.astra_db)
        self.memory_store = MemoryStore(self.db_connector)
        self.context_assembler = ContextAssembler(
            max_tokens=config.memory.max_context_tokens,
            max_memories=config.memory.max_memories_per_context
        )
        
        # Memory cache for fast access
        self.memory_cache: Dict[str, List[MemoryItem]] = {}
        self.cache_expiry: Dict[str, datetime] = {}
        self.cache_timeout = timedelta(minutes=30)
        
        # Statistics tracking
        self.stats = {
            'memories_created': 0,
            'memories_retrieved': 0,
            'context_assemblies': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    async def initialize(self):
        """Initialize the memory manager"""
        logger.info("Initializing Memory Manager...")
        
        # Initialize database connection
        await self.db_connector.connect()
        
        # Load existing gate network if available
        try:
            self.gate_network = LSTMGateNetwork.load_model("models/gate_network.pt")
            logger.info("Loaded existing gate network model")
        except FileNotFoundError:
            logger.info("No existing gate network found, using fresh model")
        
        logger.info("Memory Manager initialized successfully")
    
    async def process_new_entry(self, 
                               user_id: str,
                               feature_vector: List[float],
                               content: str,
                               embeddings: List[float],
                               metadata: Optional[Dict] = None) -> Optional[MemoryItem]:
        """
        Process a new journal entry and decide if it should become a memory
        
        Args:
            user_id: User identifier
            feature_vector: 90-dim engineered features from Component 4
            content: Original content text
            embeddings: 768-dim embeddings from Component 3
            metadata: Additional metadata
        
        Returns:
            MemoryItem if created, None if not important enough
        """
        logger.info(f"Processing new entry for user {user_id}")
        
        try:
            # Get gate decisions
            gate_decisions = self.gate_network.get_gate_decisions(feature_vector)
            
            input_decision = gate_decisions['input']
            input_score = input_decision['score']
            should_create = input_decision['decision']
            
            logger.info(f"Input gate score: {input_score:.3f}, decision: {should_create}")
            
            if should_create:
                # Create new memory
                memory = MemoryItem.create_new(
                    user_id=user_id,
                    content=content,
                    feature_vector=feature_vector,
                    initial_gate_scores={
                        'input': input_score,
                        'forget': 0.0,  # New memories don't get forgotten immediately
                        'output': 0.5   # Default output score
                    },
                    memory_type=self._classify_memory_type(content, metadata)
                )
                
                # Add metadata if provided
                if metadata:
                    memory.original_entry_id = metadata.get('entry_id')
                    memory.emotional_significance = metadata.get('emotional_significance')
                    memory.retrieval_triggers = metadata.get('triggers', [])
                
                # Save to database
                success = await self.memory_store.save_memory(memory)
                
                if success:
                    # Update cache
                    self._add_to_cache(user_id, memory)
                    self.stats['memories_created'] += 1
                    
                    logger.info(f"Created new memory: {memory.id}")
                    return memory
                else:
                    logger.error(f"Failed to save memory for user {user_id}")
                    return None
            else:
                logger.info(f"Entry not important enough to create memory (score: {input_score:.3f})")
                return None
                
        except Exception as e:
            logger.error(f"Error processing new entry for user {user_id}: {e}")
            return None
    
    async def get_relevant_context(self,
                                  user_id: str,
                                  query: str,
                                  query_features: List[float],
                                  user_context: Optional[Dict] = None,
                                  max_tokens: Optional[int] = None) -> Tuple[List[MemoryItem], Dict]:
        """
        Get relevant memory context for current conversation
        
        Args:
            user_id: User identifier
            query: Current query/message
            query_features: Feature vector for the query
            user_context: Additional user context (time, emotional state, etc.)
            max_tokens: Override default token limit
        
        Returns:
            Tuple of (selected_memories, assembly_metadata)
        """
        logger.info(f"Assembling context for user {user_id}")
        
        try:
            # Get user memories (from cache or database)
            memories = await self._get_user_memories(user_id)
            
            if not memories:
                logger.info(f"No memories found for user {user_id}")
                return [], {"reason": "no_memories_found"}
            
            # Filter memories using output gate
            relevant_memories = await self._filter_memories_by_relevance(
                memories, query_features, user_context
            )
            
            logger.info(f"Filtered to {len(relevant_memories)} relevant memories")
            
            # Assemble optimal context
            if max_tokens:
                self.context_assembler.max_tokens = max_tokens
            
            selected_memories, metadata = await self.context_assembler.assemble_context(
                relevant_memories, query_features, query, user_context
            )
            
            # Update access tracking for selected memories
            for memory in selected_memories:
                await self._update_memory_access(memory)
            
            self.stats['context_assemblies'] += 1
            self.stats['memories_retrieved'] += len(selected_memories)
            
            logger.info(f"Assembled context with {len(selected_memories)} memories")
            return selected_memories, metadata
            
        except Exception as e:
            logger.error(f"Error assembling context for user {user_id}: {e}")
            return [], {"reason": "error", "error": str(e)}
    
    async def update_memory_importance(self,
                                      memory_id: str,
                                      user_feedback: float,
                                      interaction_data: Dict):
        """
        Update memory importance based on user feedback
        
        Args:
            memory_id: Memory identifier
            user_feedback: User satisfaction score (0-1)
            interaction_data: Additional interaction context
        """
        try:
            memory = await self.memory_store.get_memory(memory_id)
            if not memory:
                logger.warning(f"Memory {memory_id} not found for update")
                return
            
            # Adjust importance based on feedback
            feedback_weight = 0.1
            current_importance = memory.importance_score
            
            if user_feedback > 0.7:  # Positive feedback
                new_importance = min(1.0, current_importance + feedback_weight)
            elif user_feedback < 0.3:  # Negative feedback
                new_importance = max(0.1, current_importance - feedback_weight)
            else:  # Neutral feedback
                new_importance = current_importance
            
            memory.importance_score = new_importance
            
            # Update gate scores if needed
            if 'context_relevance' in interaction_data:
                memory.gate_scores['output'] = interaction_data['context_relevance']
            
            # Save updates
            await self.memory_store.update_memory_scores(
                memory_id, memory.gate_scores, memory.importance_score
            )
            
            # Update cache
            self._update_cache_memory(memory)
            
            logger.info(f"Updated memory {memory_id} importance: {current_importance:.3f} -> {new_importance:.3f}")
            
        except Exception as e:
            logger.error(f"Error updating memory {memory_id}: {e}")
    
    async def run_memory_decay(self, user_id: Optional[str] = None):
        """
        Run memory decay process for a user or all users
        
        Args:
            user_id: Specific user ID, or None for all users
        """
        logger.info(f"Running memory decay for {'user ' + user_id if user_id else 'all users'}")
        
        try:
            if user_id:
                await self._decay_user_memories(user_id)
            else:
                # Get all users with memories
                user_ids = await self.memory_store.get_all_user_ids()
                
                # Process in batches to avoid overwhelming the system
                batch_size = 10
                for i in range(0, len(user_ids), batch_size):
                    batch = user_ids[i:i + batch_size]
                    
                    # Process batch concurrently
                    tasks = [self._decay_user_memories(uid) for uid in batch]
                    await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Small delay between batches
                    await asyncio.sleep(0.1)
            
            logger.info("Memory decay completed")
            
        except Exception as e:
            logger.error(f"Error running memory decay: {e}")
    
    async def _decay_user_memories(self, user_id: str):
        """Apply decay to all memories for a specific user"""
        try:
            memories = await self.memory_store.get_user_memories(user_id)
            
            updates = []
            for memory in memories:
                # Calculate forget gate score
                forget_decisions = self.gate_network.get_gate_decisions(memory.feature_vector)
                forget_score = forget_decisions['forget']['score']
                
                # Apply decay
                memory.apply_decay(
                    time_decay_rate=self.config.memory.time_decay_rate,
                    access_decay_rate=self.config.memory.access_decay_rate
                )
                
                # Update forget score
                memory.gate_scores['forget'] = forget_score
                
                # Mark for update if importance changed significantly
                updates.append((memory.id, memory.gate_scores, memory.importance_score))
            
            # Batch update database
            if updates:
                await self.memory_store.batch_update_scores(updates)
                
                # Clear cache for user to force refresh
                self._clear_user_cache(user_id)
            
        except Exception as e:
            logger.error(f"Error decaying memories for user {user_id}: {e}")
    
    async def _get_user_memories(self, user_id: str, force_refresh: bool = False) -> List[MemoryItem]:
        """Get user memories from cache or database"""
        
        # Check cache first
        if not force_refresh and self._is_cache_valid(user_id):
            self.stats['cache_hits'] += 1
            return self.memory_cache[user_id]
        
        # Load from database
        self.stats['cache_misses'] += 1
        memories = await self.memory_store.get_user_memories(user_id)
        
        # Update cache
        self.memory_cache[user_id] = memories
        self.cache_expiry[user_id] = datetime.now() + self.cache_timeout
        
        return memories
    
    async def _filter_memories_by_relevance(self,
                                           memories: List[MemoryItem],
                                           query_features: List[float],
                                           user_context: Optional[Dict]) -> List[MemoryItem]:
        """Filter memories using output gate for relevance"""
        relevant_memories = []
        
        # Create context vector for output gate if available
        context_vector = None
        if user_context:
            context_vector = self._create_context_vector(user_context)
        
        for memory in memories:
            # Get output gate decision
            output_decisions = self.gate_network.get_gate_decisions(
                memory.feature_vector, context_vector
            )
            
            output_score = output_decisions['output']['score']
            is_relevant = output_decisions['output']['decision']
            
            if is_relevant:
                # Update output score in memory
                memory.gate_scores['output'] = output_score
                relevant_memories.append(memory)
        
        return relevant_memories
    
    def _create_context_vector(self, user_context: Dict) -> List[float]:
        """Create context vector from user context"""
        context_features = []
        
        # Time features
        if 'current_time' in user_context:
            time_data = user_context['current_time']
            context_features.extend([
                time_data.get('hour', 12) / 24.0,  # Normalize hour
                float(time_data.get('is_weekend', False)),
                float(time_data.get('is_evening', False))
            ])
        else:
            context_features.extend([0.5, 0.0, 0.0])
        
        # Emotional state features
        if 'emotions' in user_context:
            emotions = user_context['emotions']
            context_features.extend([
                emotions.get('joy', 0.5),
                emotions.get('sadness', 0.5),
                emotions.get('anger', 0.5),
                emotions.get('fear', 0.5)
            ])
        else:
            context_features.extend([0.5, 0.5, 0.5, 0.5])
        
        return context_features
    
    def _classify_memory_type(self, content: str, metadata: Optional[Dict] = None) -> str:
        """Classify memory type based on content and metadata"""
        content_lower = content.lower()
        
        # Use metadata if available
        if metadata and 'memory_type' in metadata:
            return metadata['memory_type']
        
        # Simple keyword-based classification
        if any(word in content_lower for word in ['feel', 'emotion', 'sad', 'happy', 'angry', 'excited', 'worried']):
            return 'emotion'
        elif any(word in content_lower for word in ['meeting', 'appointment', 'deadline', 'event', 'conference', 'interview']):
            return 'event'
        elif any(word in content_lower for word in ['learned', 'realized', 'understand', 'insight', 'discovery', 'epiphany']):
            return 'insight'
        else:
            return 'conversation'
    
    async def _update_memory_access(self, memory: MemoryItem):
        """Update memory access tracking"""
        memory.update_access()
        await self.memory_store.update_access_tracking(memory.id, memory.last_accessed, memory.access_frequency)
    
    def _add_to_cache(self, user_id: str, memory: MemoryItem):
        """Add memory to cache"""
        if user_id not in self.memory_cache:
            self.memory_cache[user_id] = []
        
        self.memory_cache[user_id].append(memory)
        self.cache_expiry[user_id] = datetime.now() + self.cache_timeout
    
    def _update_cache_memory(self, updated_memory: MemoryItem):
        """Update a memory in cache"""
        user_id = updated_memory.user_id
        if user_id in self.memory_cache:
            for i, memory in enumerate(self.memory_cache[user_id]):
                if memory.id == updated_memory.id:
                    self.memory_cache[user_id][i] = updated_memory
                    break
    
    def _is_cache_valid(self, user_id: str) -> bool:
        """Check if cache is valid for user"""
        return (user_id in self.memory_cache and 
                user_id in self.cache_expiry and
                datetime.now() < self.cache_expiry[user_id])
    
    def _clear_user_cache(self, user_id: str):
        """Clear cache for specific user"""
        if user_id in self.memory_cache:
            del self.memory_cache[user_id]
        if user_id in self.cache_expiry:
            del self.cache_expiry[user_id]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics"""
        return {
            **self.stats,
            'cached_users': len(self.memory_cache),
            'gate_thresholds': self.gate_network.get_thresholds(),
            'config': self.config.to_dict()
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up Memory Manager...")
        
        # Save gate network
        try:
            self.gate_network.save_model("models/gate_network.pt")
            logger.info("Saved gate network model")
        except Exception as e:
            logger.error(f"Error saving gate network: {e}")
        
        # Close database connection
        await self.db_connector.close()
        
        # Clear caches
        self.memory_cache.clear()
        self.cache_expiry.clear()
        
        logger.info("Memory Manager cleanup completed")