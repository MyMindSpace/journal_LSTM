# models/memory_item.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
import uuid
import json

@dataclass
class MemoryItem:
    """Memory item structure matching Astra DB schema"""
    id: str
    user_id: str
    content_summary: str
    feature_vector: List[float]  # 90-dim engineered features
    gate_scores: Dict[str, float]
    importance_score: float
    last_accessed: datetime
    created_at: datetime
    access_frequency: int
    memory_type: str
    
    # Optional metadata
    original_entry_id: Optional[str] = None
    emotional_significance: Optional[float] = None
    temporal_relevance: Optional[float] = None
    relationships: List[str] = field(default_factory=list)
    context_needed: Dict[str, Any] = field(default_factory=dict)
    retrieval_triggers: List[str] = field(default_factory=list)
    
    @classmethod
    def create_new(cls, user_id: str, content: str, feature_vector: List[float],
                   initial_gate_scores: Dict[str, float], memory_type: str = "conversation"):
        """Create a new memory item with timezone-aware datetimes"""
        now = datetime.now(timezone.utc)
        return cls(
            id=str(uuid.uuid4()),
            user_id=user_id,
            content_summary=content[:500],  # Limit summary length
            feature_vector=feature_vector,
            gate_scores=initial_gate_scores,
            importance_score=initial_gate_scores.get('input', 0.5),
            last_accessed=now,
            created_at=now,
            access_frequency=0,
            memory_type=memory_type
        )
    
    def update_access(self):
        """Update access tracking with timezone-aware datetime"""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_frequency += 1
    
    def update_gate_scores(self, new_scores: Dict[str, float]):
        """Update gate scores"""
        self.gate_scores.update(new_scores)
        self.importance_score = new_scores.get('input', self.importance_score)
        self.last_accessed = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'memory_type': self.memory_type,
            'content_summary': self.content_summary,
            'original_entry_id': self.original_entry_id,
            'importance_score': self.importance_score,
            'emotional_significance': self.emotional_significance,
            'temporal_relevance': self.temporal_relevance,
            'access_frequency': self.access_frequency,
            'last_accessed': self.last_accessed,
            'created_at': self.created_at,
            'gate_scores': self.gate_scores,
            'feature_vector': self.feature_vector,
            'relationships': self.relationships,
            'context_needed': self.context_needed,
            'retrieval_triggers': self.retrieval_triggers
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create from dictionary (from database)"""
        # Handle datetime parsing - more robust version
        created_at = data['created_at']
        if isinstance(created_at, str):
            # Handle various ISO string formats
            try:
                if created_at.endswith('Z'):
                    created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                elif '+' in created_at or created_at.endswith('00:00'):
                    created_at = datetime.fromisoformat(created_at)
                else:
                    # Assume UTC if no timezone info
                    created_at = datetime.fromisoformat(created_at).replace(tzinfo=timezone.utc)
            except ValueError:
                # Fallback to current time if parsing fails
                created_at = datetime.now(timezone.utc)
        elif isinstance(created_at, datetime) and created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)
            
        last_accessed = data['last_accessed']
        if isinstance(last_accessed, str):
            try:
                if last_accessed.endswith('Z'):
                    last_accessed = datetime.fromisoformat(last_accessed.replace('Z', '+00:00'))
                elif '+' in last_accessed or last_accessed.endswith('00:00'):
                    last_accessed = datetime.fromisoformat(last_accessed)
                else:
                    # Assume UTC if no timezone info
                    last_accessed = datetime.fromisoformat(last_accessed).replace(tzinfo=timezone.utc)
            except ValueError:
                # Fallback to current time if parsing fails
                last_accessed = datetime.now(timezone.utc)
        elif isinstance(last_accessed, datetime) and last_accessed.tzinfo is None:
            last_accessed = last_accessed.replace(tzinfo=timezone.utc)
        elif last_accessed is None:
            last_accessed = datetime.now(timezone.utc)
        
        # Handle JSON parsing for complex fields
        gate_scores = data.get('gate_scores', {})
        if isinstance(gate_scores, str):
            try:
                gate_scores = json.loads(gate_scores)
            except (json.JSONDecodeError, TypeError):
                gate_scores = {}
        
        context_needed = data.get('context_needed', {})
        if isinstance(context_needed, str):
            try:
                context_needed = json.loads(context_needed)
            except (json.JSONDecodeError, TypeError):
                context_needed = {}
        
        return cls(
            id=data['id'],
            user_id=data['user_id'],
            content_summary=data['content_summary'],
            feature_vector=data['feature_vector'],
            gate_scores=gate_scores,
            importance_score=data['importance_score'],
            last_accessed=last_accessed,
            created_at=created_at,
            access_frequency=data['access_frequency'],
            memory_type=data['memory_type'],
            original_entry_id=data.get('original_entry_id'),
            emotional_significance=data.get('emotional_significance'),
            temporal_relevance=data.get('temporal_relevance'),
            relationships=data.get('relationships', []),
            context_needed=context_needed,
            retrieval_triggers=data.get('retrieval_triggers', [])
        )
    
    def get_relevance_score(self, query_vector: List[float]) -> float:
        """Calculate relevance to query vector"""
        from utils.similarity import cosine_similarity
        
        # Use feature vector for similarity calculation
        similarity = cosine_similarity(self.feature_vector, query_vector)
        
        # Weight by importance and recency
        now = datetime.now(timezone.utc)
        recency_days = (now - self.last_accessed).days
        recency_weight = max(0.1, 1.0 - (recency_days * 0.05))  # Decay over 20 days
        
        return similarity * self.importance_score * recency_weight
    
    def __str__(self) -> str:
        return f"Memory({self.id[:8]}, {self.memory_type}, {self.importance_score:.2f})"
    
    def __repr__(self) -> str:
        return self.__str__()