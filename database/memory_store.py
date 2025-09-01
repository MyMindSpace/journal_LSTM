# database/memory_store.py
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import uuid
from dataclasses import asdict

from database.astra_connector import AstraDBConnector
from models.memory_item import MemoryItem
from utils.similarity import SimilarityCalculator
from utils.token_counter import estimate_tokens

logger = logging.getLogger(__name__)

class MemoryStore:
    """
    High-level memory storage and retrieval operations
    Provides business logic layer over AstraDBConnector
    """
    
    def __init__(self, db_connector: AstraDBConnector):
        self.db = db_connector
        self.similarity_calculator = SimilarityCalculator(cache_size=1000)
        
        # Performance tracking
        self.operation_stats = {
            'memories_saved': 0,
            'memories_retrieved': 0,
            'similarity_searches': 0,
            'batch_operations': 0,
            'cache_operations': 0,
            'errors': 0
        }
        
        # Memory cache for recently accessed items
        self.memory_cache = {}
        self.cache_expiry = {}
        self.cache_ttl = timedelta(minutes=30)
    
    async def save_memory(self, memory: MemoryItem) -> bool:
        """
        Save memory item to database with validation
        
        Args:
            memory: MemoryItem to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate memory item
            if not self._validate_memory(memory):
                logger.error(f"Invalid memory item: {memory.id}")
                return False
            
            # Convert to database format
            memory_data = self._memory_to_db_format(memory)
            
            # Save to database
            success = await self.db.save_memory(memory_data)
            
            if success:
                # Update cache
                self._add_to_cache(memory)
                self.operation_stats['memories_saved'] += 1
                logger.info(f"Saved memory {memory.id} for user {memory.user_id}")
            else:
                self.operation_stats['errors'] += 1
                logger.error(f"Failed to save memory {memory.id}")
            
            return success
            
        except Exception as e:
            self.operation_stats['errors'] += 1
            logger.error(f"Error saving memory {memory.id}: {e}")
            return False
    
    async def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """
        Get specific memory by ID
        
        Args:
            memory_id: Memory identifier
            
        Returns:
            MemoryItem if found, None otherwise
        """
        try:
            # Check cache first
            if self._is_cached(memory_id):
                self.operation_stats['cache_operations'] += 1
                return self.memory_cache[memory_id]
            
            # Get from database
            memory_data = await self.db.get_memory_by_id(memory_id)
            
            if memory_data:
                memory = self._db_format_to_memory(memory_data)
                self._add_to_cache(memory)
                self.operation_stats['memories_retrieved'] += 1
                return memory
            
            return None
            
        except Exception as e:
            self.operation_stats['errors'] += 1
            logger.error(f"Error getting memory {memory_id}: {e}")
            return None
    
    async def get_user_memories(self, user_id: str, 
                               limit: Optional[int] = None,
                               memory_type: Optional[str] = None,
                               min_importance: Optional[float] = None) -> List[MemoryItem]:
        """
        Get memories for a specific user with optional filtering
        
        Args:
            user_id: User identifier
            limit: Maximum number of memories to return
            memory_type: Filter by memory type
            min_importance: Minimum importance score
            
        Returns:
            List of MemoryItem objects
        """
        try:
            # Get from database
            memory_data_list = await self.db.get_user_memories(user_id, limit or 1000)
            
            # Convert to MemoryItem objects
            memories = []
            for memory_data in memory_data_list:
                try:
                    memory = self._db_format_to_memory(memory_data)
                    
                    # Apply filters
                    if memory_type and memory.memory_type != memory_type:
                        continue
                    
                    if min_importance and memory.importance_score < min_importance:
                        continue
                    
                    memories.append(memory)
                    
                    # Add to cache for future access
                    self._add_to_cache(memory)
                    
                except Exception as e:
                    logger.warning(f"Error parsing memory data: {e}")
                    continue
            
            self.operation_stats['memories_retrieved'] += len(memories)
            logger.info(f"Retrieved {len(memories)} memories for user {user_id}")
            
            return memories
            
        except Exception as e:
            self.operation_stats['errors'] += 1
            logger.error(f"Error getting user memories for {user_id}: {e}")
            return []
    
    async def find_similar_memories(self, 
                                   user_id: str,
                                   query_vector: List[float],
                                   similarity_threshold: float = 0.7,
                                   top_k: int = 10,
                                   memory_types: Optional[List[str]] = None) -> List[Tuple[MemoryItem, float]]:
        """
        Find memories similar to query vector using semantic search
        
        Args:
            user_id: User identifier
            query_vector: Feature vector to search for
            similarity_threshold: Minimum similarity score
            top_k: Maximum number of results
            memory_types: Filter by memory types
            
        Returns:
            List of (MemoryItem, similarity_score) tuples
        """
        try:
            # Get user memories
            memories = await self.get_user_memories(user_id)
            
            if not memories:
                return []
            
            # Filter by memory types if specified
            if memory_types:
                memories = [m for m in memories if m.memory_type in memory_types]
            
            # Calculate similarities
            similar_memories = []
            
            for memory in memories:
                try:
                    similarity = self.similarity_calculator.calculate(
                        query_vector, 
                        memory.feature_vector,
                        method='cosine'
                    )
                    
                    if similarity >= similarity_threshold:
                        similar_memories.append((memory, similarity))
                        
                except Exception as e:
                    logger.warning(f"Error calculating similarity for memory {memory.id}: {e}")
                    continue
            
            # Sort by similarity and return top_k
            similar_memories.sort(key=lambda x: x[1], reverse=True)
            result = similar_memories[:top_k]
            
            self.operation_stats['similarity_searches'] += 1
            logger.info(f"Found {len(result)} similar memories for user {user_id}")
            
            return result
            
        except Exception as e:
            self.operation_stats['errors'] += 1
            logger.error(f"Error finding similar memories for {user_id}: {e}")
            return []
    
    async def update_memory_scores(self, memory_id: str, gate_scores: Dict[str, float], importance_score: float):
        """
        Update memory gate scores and importance
        
        Args:
            memory_id: Memory identifier
            gate_scores: Updated gate scores
            importance_score: Updated importance score
        """
        try:
            await self.db.update_memory_scores(memory_id, gate_scores, importance_score)
            
            # Update cache if present
            if memory_id in self.memory_cache:
                memory = self.memory_cache[memory_id]
                memory.gate_scores.update(gate_scores)
                memory.importance_score = importance_score
            
            logger.debug(f"Updated scores for memory {memory_id}")
            
        except Exception as e:
            self.operation_stats['errors'] += 1
            logger.error(f"Error updating memory scores for {memory_id}: {e}")
            raise
    
    async def update_access_tracking(self, memory_id: str, last_accessed: datetime, access_frequency: int):
        """
        Update memory access tracking information
        
        Args:
            memory_id: Memory identifier
            last_accessed: Last access timestamp
            access_frequency: Updated access frequency
        """
        try:
            await self.db.update_access_tracking(memory_id, last_accessed, access_frequency)
            
            # Update cache if present
            if memory_id in self.memory_cache:
                memory = self.memory_cache[memory_id]
                memory.last_accessed = last_accessed
                memory.access_frequency = access_frequency
            
            logger.debug(f"Updated access tracking for memory {memory_id}")
            
        except Exception as e:
            self.operation_stats['errors'] += 1
            logger.error(f"Error updating access tracking for {memory_id}: {e}")
            raise
    
    async def batch_update_scores(self, updates: List[Tuple[str, Dict[str, float], float]]):
        """
        Batch update multiple memory scores for efficiency
        
        Args:
            updates: List of (memory_id, gate_scores, importance_score) tuples
        """
        try:
            await self.db.batch_update_scores(updates)
            
            # Update cache entries
            for memory_id, gate_scores, importance_score in updates:
                if memory_id in self.memory_cache:
                    memory = self.memory_cache[memory_id]
                    memory.gate_scores.update(gate_scores)
                    memory.importance_score = importance_score
            
            self.operation_stats['batch_operations'] += 1
            logger.info(f"Batch updated {len(updates)} memory scores")
            
        except Exception as e:
            self.operation_stats['errors'] += 1
            logger.error(f"Error batch updating memory scores: {e}")
            raise
    
    async def delete_memory(self, memory_id: str, user_id: str) -> bool:
        """
        Delete a memory (mark as deleted or remove completely)
        
        Args:
            memory_id: Memory identifier
            user_id: User identifier for security
            
        Returns:
            True if successful
        """
        try:
            # For now, we'll set importance to 0 instead of hard delete
            # This preserves the memory for potential recovery
            await self.update_memory_scores(memory_id, {'forget': 1.0}, 0.0)
            
            # Remove from cache
            if memory_id in self.memory_cache:
                del self.memory_cache[memory_id]
                if memory_id in self.cache_expiry:
                    del self.cache_expiry[memory_id]
            
            logger.info(f"Soft deleted memory {memory_id} for user {user_id}")
            return True
            
        except Exception as e:
            self.operation_stats['errors'] += 1
            logger.error(f"Error deleting memory {memory_id}: {e}")
            return False
    
    async def get_memory_statistics(self, user_id: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics about user's memories
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with memory statistics
        """
        try:
            memories = await self.get_user_memories(user_id)
            
            if not memories:
                return {
                    'total_memories': 0,
                    'by_type': {},
                    'importance_stats': {},
                    'age_stats': {},
                    'access_stats': {}
                }
            
            # Calculate statistics
            by_type = {}
            importance_scores = []
            ages_days = []
            access_frequencies = []
            total_tokens = 0
            
            now = datetime.now()
            
            for memory in memories:
                # Type distribution
                memory_type = memory.memory_type
                by_type[memory_type] = by_type.get(memory_type, 0) + 1
                
                # Importance scores
                importance_scores.append(memory.importance_score)
                
                # Age in days
                age_days = (now - memory.created_at).days
                ages_days.append(age_days)
                
                # Access frequency
                access_frequencies.append(memory.access_frequency)
                
                # Token count
                tokens = estimate_tokens(memory.content_summary)
                total_tokens += tokens
            
            return {
                'total_memories': len(memories),
                'by_type': by_type,
                'importance_stats': {
                    'mean': sum(importance_scores) / len(importance_scores),
                    'min': min(importance_scores),
                    'max': max(importance_scores),
                    'above_threshold': sum(1 for s in importance_scores if s > 0.7)
                },
                'age_stats': {
                    'mean_days': sum(ages_days) / len(ages_days),
                    'oldest_days': max(ages_days),
                    'newest_days': min(ages_days),
                    'recent_count': sum(1 for age in ages_days if age <= 7)
                },
                'access_stats': {
                    'mean_frequency': sum(access_frequencies) / len(access_frequencies),
                    'most_accessed': max(access_frequencies),
                    'never_accessed': sum(1 for freq in access_frequencies if freq == 0)
                },
                'token_stats': {
                    'total_tokens': total_tokens,
                    'average_tokens_per_memory': total_tokens / len(memories)
                }
            }
            
        except Exception as e:
            self.operation_stats['errors'] += 1
            logger.error(f"Error getting memory statistics for {user_id}: {e}")
            return {}
    
    async def cleanup_old_memories(self, user_id: str, 
                                  older_than_days: int = 90,
                                  min_importance: float = 0.1,
                                  max_to_delete: int = 100) -> int:
        """
        Cleanup old, unimportant memories
        
        Args:
            user_id: User identifier
            older_than_days: Delete memories older than this
            min_importance: Only delete memories with importance below this
            max_to_delete: Maximum number of memories to delete in one operation
            
        Returns:
            Number of memories deleted
        """
        try:
            memories = await self.get_user_memories(user_id)
            cutoff_date = datetime.now() - timedelta(days=older_than_days)
            
            # Find candidates for deletion
            candidates = []
            for memory in memories:
                if (memory.created_at < cutoff_date and 
                    memory.importance_score < min_importance and
                    memory.access_frequency == 0):  # Never accessed
                    candidates.append(memory)
            
            # Sort by importance (lowest first) and limit
            candidates.sort(key=lambda m: m.importance_score)
            to_delete = candidates[:max_to_delete]
            
            # Delete memories
            deleted_count = 0
            for memory in to_delete:
                if await self.delete_memory(memory.id, user_id):
                    deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old memories for user {user_id}")
            return deleted_count
            
        except Exception as e:
            self.operation_stats['errors'] += 1
            logger.error(f"Error cleaning up memories for {user_id}: {e}")
            return 0
    
    async def get_all_user_ids(self) -> List[str]:
        """Get all unique user IDs"""
        try:
            return await self.db.get_all_user_ids()
        except Exception as e:
            self.operation_stats['errors'] += 1
            logger.error(f"Error getting all user IDs: {e}")
            return []
    
    # Helper methods
    def _validate_memory(self, memory: MemoryItem) -> bool:
        """Validate memory item before saving"""
        if not memory.id or not memory.user_id:
            return False
        
        if not memory.content_summary or not memory.content_summary.strip():
            return False
        
        if not memory.feature_vector or len(memory.feature_vector) != 90:
            return False
        
        if not isinstance(memory.gate_scores, dict):
            return False
        
        if not (0 <= memory.importance_score <= 1):
            return False
        
        return True
    
    def _memory_to_db_format(self, memory: MemoryItem) -> Dict[str, Any]:
        """Convert MemoryItem to database format"""
        return {
            'id': memory.id,
            'user_id': memory.user_id,
            'memory_type': memory.memory_type,
            'content_summary': memory.content_summary,
            'original_entry_id': memory.original_entry_id,
            'importance_score': memory.importance_score,
            'emotional_significance': memory.emotional_significance,
            'temporal_relevance': memory.temporal_relevance,
            'access_frequency': memory.access_frequency,
            'last_accessed': memory.last_accessed,
            'created_at': memory.created_at,
            'gate_scores': memory.gate_scores,
            'feature_vector': memory.feature_vector,
            'relationships': memory.relationships,
            'context_needed': memory.context_needed,
            'retrieval_triggers': memory.retrieval_triggers
        }
    
    def _db_format_to_memory(self, data: Dict[str, Any]) -> MemoryItem:
        """Convert database format to MemoryItem"""
        # Parse JSON fields
        gate_scores = json.loads(data.get('gate_scores', '{}')) if isinstance(data.get('gate_scores'), str) else data.get('gate_scores', {})
        context_needed = json.loads(data.get('context_needed', '{}')) if isinstance(data.get('context_needed'), str) else data.get('context_needed', {})
        
        return MemoryItem(
            id=data['id'],
            user_id=data['user_id'],
            content_summary=data['content_summary'],
            feature_vector=data['feature_vector'],
            gate_scores=gate_scores,
            importance_score=data['importance_score'],
            last_accessed=data['last_accessed'],
            created_at=data['created_at'],
            access_frequency=data['access_frequency'],
            memory_type=data['memory_type'],
            original_entry_id=data.get('original_entry_id'),
            emotional_significance=data.get('emotional_significance'),
            temporal_relevance=data.get('temporal_relevance'),
            relationships=data.get('relationships', []),
            context_needed=context_needed,
            retrieval_triggers=data.get('retrieval_triggers', [])
        )
    
    def _add_to_cache(self, memory: MemoryItem):
        """Add memory to cache"""
        self.memory_cache[memory.id] = memory
        self.cache_expiry[memory.id] = datetime.now() + self.cache_ttl
        
        # Simple cache size management
        if len(self.memory_cache) > 1000:
            # Remove oldest entries
            oldest_items = sorted(self.cache_expiry.items(), key=lambda x: x[1])[:100]
            for memory_id, _ in oldest_items:
                if memory_id in self.memory_cache:
                    del self.memory_cache[memory_id]
                del self.cache_expiry[memory_id]
    
    def _is_cached(self, memory_id: str) -> bool:
        """Check if memory is in cache and not expired"""
        if memory_id not in self.memory_cache:
            return False
        
        if memory_id not in self.cache_expiry:
            return False
        
        if datetime.now() > self.cache_expiry[memory_id]:
            # Remove expired entry
            del self.memory_cache[memory_id]
            del self.cache_expiry[memory_id]
            return False
        
        return True
    
    def get_operation_stats(self) -> Dict[str, Any]:
        """Get detailed operation statistics"""
        cache_stats = self.similarity_calculator.get_cache_stats()
        
        return {
            'memory_operations': self.operation_stats,
            'similarity_cache': cache_stats,
            'memory_cache_size': len(self.memory_cache),
            'memory_cache_hit_rate': self._calculate_cache_hit_rate()
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate memory cache hit rate"""
        total_cache_ops = self.operation_stats['cache_operations']
        total_retrievals = self.operation_stats['memories_retrieved']
        
        if total_retrievals + total_cache_ops == 0:
            return 0.0
        
        return total_cache_ops / (total_retrievals + total_cache_ops)
    
    def clear_cache(self):
        """Clear memory cache"""
        self.memory_cache.clear()
        self.cache_expiry.clear()
        self.similarity_calculator.clear_cache()
        logger.info("Cleared memory store caches")