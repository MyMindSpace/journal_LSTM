# models/memory_item.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
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
        """Create a new memory item"""
        return cls(
            id=str(uuid.uuid4()),
            user_id=user_id,
            content_summary=content[:500],  # Limit summary length
            feature_vector=feature_vector,
            gate_scores=initial_gate_scores,
            importance_score=initial_gate_scores.get('input', 0.5),
            last_accessed=datetime.now(),
            created_at=datetime.now(),
            access_frequency=0,
            memory_type=memory_type
        )
    
    def update_access(self):
        """Update access tracking"""
        self.last_accessed = datetime.now()
        self.access_frequency += 1
    
    def update_gate_scores(self, new_scores: Dict[str, float]):
        """Update gate scores"""
        self.gate_scores.update(new_scores)
        # Update importance based on input score
        if 'input' in new_scores:
            self.importance_score = new_scores['input']
    
    def apply_decay(self, time_decay_rate: float, access_decay_rate: float):
        """Apply time-based decay to importance"""
        days_old = (datetime.now() - self.created_at).days
        days_since_access = (datetime.now() - self.last_accessed).days
        
        time_decay = max(0.1, 1.0 - (days_old * time_decay_rate))
        access_decay = max(0.1, 1.0 - (days_since_access * access_decay_rate))
        
        # Apply forget gate influence
        forget_score = self.gate_scores.get('forget', 0.0)
        self.importance_score *= time_decay * access_decay * (1 - forget_score)
        
        # Ensure importance doesn't go below threshold
        self.importance_score = max(0.01, self.importance_score)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'content_summary': self.content_summary,
            'feature_vector': self.feature_vector,
            'gate_scores': json.dumps(self.gate_scores),
            'importance_score': self.importance_score,
            'last_accessed': self.last_accessed,
            'created_at': self.created_at,
            'access_frequency': self.access_frequency,
            'memory_type': self.memory_type,
            'original_entry_id': self.original_entry_id,
            'emotional_significance': self.emotional_significance,
            'temporal_relevance': self.temporal_relevance,
            'relationships': self.relationships,
            'context_needed': json.dumps(self.context_needed),
            'retrieval_triggers': self.retrieval_triggers
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create from dictionary (from database)"""
        return cls(
            id=data['id'],
            user_id=data['user_id'],
            content_summary=data['content_summary'],
            feature_vector=data['feature_vector'],
            gate_scores=json.loads(data.get('gate_scores', '{}')),
            importance_score=data['importance_score'],
            last_accessed=data['last_accessed'],
            created_at=data['created_at'],
            access_frequency=data['access_frequency'],
            memory_type=data['memory_type'],
            original_entry_id=data.get('original_entry_id'),
            emotional_significance=data.get('emotional_significance'),
            temporal_relevance=data.get('temporal_relevance'),
            relationships=data.get('relationships', []),
            context_needed=json.loads(data.get('context_needed', '{}')),
            retrieval_triggers=data.get('retrieval_triggers', [])
        )
    
    def get_relevance_score(self, query_vector: List[float]) -> float:
        """Calculate relevance to query vector"""
        from ..utils.similarity import cosine_similarity
        
        # Use feature vector for similarity calculation
        similarity = cosine_similarity(self.feature_vector, query_vector)
        
        # Weight by importance and recency
        recency_days = (datetime.now() - self.last_accessed).days
        recency_weight = max(0.1, 1.0 - (recency_days * 0.05))  # Decay over 20 days
        
        return similarity * self.importance_score * recency_weight
    
    def __str__(self) -> str:
        return f"Memory({self.id[:8]}, {self.memory_type}, {self.importance_score:.2f})"
    
    def __repr__(self) -> str:
        return self.__str__()