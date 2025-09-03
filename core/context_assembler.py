# core/context_assembler.py
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta, timezone
import numpy as np
import logging

from models.memory_item import MemoryItem
from utils.similarity import cosine_similarity
from utils.token_counter import estimate_tokens

logger = logging.getLogger(__name__)

class ContextAssembler:
    """Assembles relevant memory context for conversations"""
    
    def __init__(self, max_tokens: int = 2000, max_memories: int = 20):
        self.max_tokens = max_tokens
        self.max_memories = max_memories
        
        # Diversity weights for memory selection
        self.diversity_weights = {
            'conversation': 0.4,
            'event': 0.25,
            'emotion': 0.25,
            'insight': 0.1
        }
    
    def _ensure_datetime(self, dt_value) -> datetime:
        """Ensure a value is a datetime object - FIXED for string handling"""
        if isinstance(dt_value, str):
            try:
                if dt_value.endswith('Z'):
                    return datetime.fromisoformat(dt_value.replace('Z', '+00:00'))
                elif '+' in dt_value or dt_value.endswith('00:00'):
                    return datetime.fromisoformat(dt_value)
                else:
                    # Assume UTC if no timezone info
                    return datetime.fromisoformat(dt_value).replace(tzinfo=timezone.utc)
            except ValueError:
                # Fallback to current time if parsing fails
                return datetime.now(timezone.utc)
        elif isinstance(dt_value, datetime):
            if dt_value.tzinfo is None:
                return dt_value.replace(tzinfo=timezone.utc)
            return dt_value
        else:
            # Fallback for any other type
            return datetime.now(timezone.utc)
    
    async def assemble_context(self, 
                              memories: List[MemoryItem],
                              query_vector: List[float],
                              current_query: str,
                              user_context: Optional[Dict] = None) -> Tuple[List[MemoryItem], Dict]:
        """
        Assemble optimal context from available memories
        
        Returns:
            Tuple of (selected_memories, assembly_metadata)
        """
        if not memories:
            return [], {"reason": "no_memories_available", "token_count": 0}
        
        try:
            # Step 1: Score all memories for relevance
            scored_memories = await self._score_memories(memories, query_vector, current_query, user_context)
            
            # Step 2: Apply diversity filtering
            diverse_memories = self._ensure_diversity(scored_memories)
            
            # Step 3: Fit within token budget
            selected_memories, metadata = self._fit_token_budget(diverse_memories, current_query)
            
            # Step 4: Update access tracking
            for memory in selected_memories:
                memory.update_access()
            
            return selected_memories, metadata
            
        except Exception as e:
            logger.error(f"Error in assemble_context: {e}")
            return [], {"reason": "error", "error": str(e)}
    
    async def _score_memories(self, 
                             memories: List[MemoryItem],
                             query_vector: List[float],
                             current_query: str,
                             user_context: Optional[Dict] = None) -> List[Tuple[MemoryItem, float]]:
        """Score memories for relevance to current query"""
        scored_memories = []
        
        for memory in memories:
            try:
                # Base similarity score
                similarity = cosine_similarity(memory.feature_vector, query_vector)
                
                # Output gate score (relevance for current context)
                output_score = memory.gate_scores.get('output', 0.5)
                
                # Importance score
                importance = memory.importance_score
                
                # Recency boost - FIXED to handle string datetimes
                recency_score = self._calculate_recency_score(memory)
                
                # Access frequency (popular memories get slight boost)
                frequency_boost = min(0.2, memory.access_frequency * 0.01)
                
                # Temporal relevance (if user context provides time info)
                temporal_score = 1.0
                if user_context and 'current_time_context' in user_context:
                    temporal_score = self._calculate_temporal_relevance(memory, user_context)
                
                # Final weighted score
                final_score = (
                    similarity * 0.35 +
                    output_score * 0.25 +
                    importance * 0.20 +
                    recency_score * 0.10 +
                    frequency_boost +
                    temporal_score * 0.10
                )
                
                scored_memories.append((memory, final_score))
                
            except Exception as e:
                logger.warning(f"Error scoring memory {memory.id}: {e}")
                continue
        
        # Sort by score descending
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        return scored_memories
    
    def _calculate_recency_score(self, memory: MemoryItem) -> float:
        """Calculate recency score (more recent = higher score) - FIXED for datetime handling"""
        now = datetime.now(timezone.utc)
        
        # Ensure datetime objects - FIXED
        created_at = self._ensure_datetime(memory.created_at)
        last_accessed = self._ensure_datetime(memory.last_accessed)
        
        try:
            days_old = (now - created_at).days
            days_since_access = (now - last_accessed).days
            
            # Recent creation boost
            creation_recency = max(0.0, 1.0 - (days_old * 0.03))  # Decay over ~30 days
            
            # Recent access boost
            access_recency = max(0.0, 1.0 - (days_since_access * 0.05))  # Decay over ~20 days
            
            return (creation_recency + access_recency) / 2
            
        except Exception as e:
            logger.warning(f"Error calculating recency for memory {memory.id}: {e}")
            return 0.5  # Default middle score
    
    def _calculate_temporal_relevance(self, memory: MemoryItem, user_context: Dict) -> float:
        """Calculate how temporally relevant this memory is"""
        current_time = user_context.get('current_time_context', {})
        
        # Ensure datetime object
        created_at = self._ensure_datetime(memory.created_at)
        
        try:
            # Check for time-of-day relevance
            if 'hour' in current_time:
                memory_hour = created_at.hour
                current_hour = current_time['hour']
                hour_diff = min(abs(memory_hour - current_hour), 24 - abs(memory_hour - current_hour))
                time_relevance = max(0.0, 1.0 - (hour_diff / 12))  # Similar times get boost
            else:
                time_relevance = 0.5
            
            # Check for day-of-week relevance
            if 'weekday' in current_time:
                memory_weekday = created_at.weekday()
                current_weekday = current_time['weekday']
                if memory_weekday == current_weekday:
                    weekday_relevance = 1.0
                elif abs(memory_weekday - current_weekday) <= 1:
                    weekday_relevance = 0.7
                else:
                    weekday_relevance = 0.3
            else:
                weekday_relevance = 0.5
            
            return (time_relevance + weekday_relevance) / 2
            
        except Exception as e:
            logger.warning(f"Error calculating temporal relevance for memory {memory.id}: {e}")
            return 0.5  # Default middle score
    
    def _ensure_diversity(self, scored_memories: List[Tuple[MemoryItem, float]]) -> List[Tuple[MemoryItem, float]]:
        """Ensure diversity in memory types while maintaining relevance"""
        if len(scored_memories) <= self.max_memories:
            return scored_memories
        
        # Group by memory type
        type_groups = {}
        for memory, score in scored_memories:
            mem_type = memory.memory_type
            if mem_type not in type_groups:
                type_groups[mem_type] = []
            type_groups[mem_type].append((memory, score))
        
        # Calculate target count for each type
        diverse_memories = []
        remaining_slots = self.max_memories
        
        for mem_type, weight in self.diversity_weights.items():
            if mem_type in type_groups and remaining_slots > 0:
                target_count = max(1, int(self.max_memories * weight))
                actual_count = min(target_count, len(type_groups[mem_type]), remaining_slots)
                
                # Take top memories of this type
                type_memories = sorted(type_groups[mem_type], key=lambda x: x[1], reverse=True)
                diverse_memories.extend(type_memories[:actual_count])
                remaining_slots -= actual_count
        
        # Fill remaining slots with highest scoring memories
        if remaining_slots > 0:
            used_memory_ids = {mem.id for mem, _ in diverse_memories}
            remaining_memories = [(mem, score) for mem, score in scored_memories 
                                 if mem.id not in used_memory_ids]
            
            remaining_memories.sort(key=lambda x: x[1], reverse=True)
            diverse_memories.extend(remaining_memories[:remaining_slots])
        
        # Sort final list by score
        diverse_memories.sort(key=lambda x: x[1], reverse=True)
        return diverse_memories
    
    def _fit_token_budget(self, scored_memories: List[Tuple[MemoryItem, float]], query: str) -> Tuple[List[MemoryItem], Dict]:
        """Select memories that fit within token budget"""
        selected_memories = []
        total_tokens = 0
        
        # Reserve tokens for query and formatting overhead
        query_tokens = estimate_tokens(query)
        overhead_tokens = 200  # For formatting, instructions, etc.
        available_tokens = self.max_tokens - query_tokens - overhead_tokens
        
        metadata = {
            "query_tokens": query_tokens,
            "overhead_tokens": overhead_tokens,
            "available_tokens": available_tokens,
            "memories_considered": len(scored_memories),
            "memories_selected": 0,
            "token_utilization": 0.0,
            "diversity_stats": {},
            "avg_relevance_score": 0.0
        }
        
        if available_tokens <= 0:
            metadata["reason"] = "insufficient_token_budget"
            return [], metadata
        
        # Track diversity
        type_counts = {}
        relevance_scores = []
        
        # Select memories that fit within token budget
        for memory, score in scored_memories:
            memory_tokens = estimate_tokens(memory.content_summary)
            
            if total_tokens + memory_tokens <= available_tokens:
                selected_memories.append(memory)
                total_tokens += memory_tokens
                relevance_scores.append(score)
                
                # Track diversity
                mem_type = memory.memory_type
                type_counts[mem_type] = type_counts.get(mem_type, 0) + 1
            else:
                # Try to fit smaller memories if this one is too big
                continue
        
        # Update metadata
        metadata.update({
            "memories_selected": len(selected_memories),
            "total_tokens_used": total_tokens,
            "token_utilization": total_tokens / self.max_tokens,
            "diversity_stats": type_counts,
            "avg_relevance_score": np.mean(relevance_scores) if relevance_scores else 0.0,
            "reason": "success" if selected_memories else "no_memories_fit_budget"
        })
        
        return selected_memories, metadata
    
    def format_context_for_llm(self, memories: List[MemoryItem], query: str) -> str:
        """Format selected memories into context string for LLM"""
        if not memories:
            return "No relevant memories found."
        
        context_parts = [
            "=== RELEVANT MEMORIES ===\n"
        ]
        
        # Group by type for better organization
        type_groups = {}
        for memory in memories:
            mem_type = memory.memory_type
            if mem_type not in type_groups:
                type_groups[mem_type] = []
            type_groups[mem_type].append(memory)
        
        # Format each type group
        for mem_type, type_memories in type_groups.items():
            if type_memories:
                context_parts.append(f"\n--- {mem_type.upper()} MEMORIES ---")
                
                for memory in type_memories:
                    # Ensure datetime for formatting
                    created_at = self._ensure_datetime(memory.created_at)
                    time_str = created_at.strftime("%Y-%m-%d %H:%M")
                    importance_str = f"{memory.importance_score:.2f}"
                    
                    context_parts.append(
                        f"\n[{time_str} | Importance: {importance_str}]\n"
                        f"{memory.content_summary}\n"
                    )
        
        context_parts.append(f"\n=== CURRENT QUERY ===\n{query}\n")
        
        return "".join(context_parts)
    
    def get_context_statistics(self, memories: List[MemoryItem]) -> Dict:
        """Get statistics about the assembled context"""
        if not memories:
            return {"total_memories": 0}
        
        type_counts = {}
        importance_scores = []
        ages_days = []
        
        now = datetime.now(timezone.utc)
        
        for memory in memories:
            # Type distribution
            mem_type = memory.memory_type
            type_counts[mem_type] = type_counts.get(mem_type, 0) + 1
            
            # Importance distribution
            importance_scores.append(memory.importance_score)
            
            # Age distribution - ensure datetime
            try:
                created_at = self._ensure_datetime(memory.created_at)
                age_days = (now - created_at).days
                ages_days.append(age_days)
            except Exception:
                ages_days.append(0)  # Default for error cases
        
        return {
            "total_memories": len(memories),
            "type_distribution": type_counts,
            "avg_importance": np.mean(importance_scores) if importance_scores else 0.0,
            "importance_range": (min(importance_scores), max(importance_scores)) if importance_scores else (0, 0),
            "avg_age_days": np.mean(ages_days) if ages_days else 0.0,
            "age_range_days": (min(ages_days), max(ages_days)) if ages_days else (0, 0),
            "newest_memory_age": min(ages_days) if ages_days else 0,
            "oldest_memory_age": max(ages_days) if ages_days else 0
        }