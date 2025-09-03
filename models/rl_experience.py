# models/rl_experience.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
import uuid
import numpy as np
import json

@dataclass
class RLState:
    """State representation for RL training"""
    user_id: str
    feature_vector: List[float]  # 90-dim engineered features
    context_vector: List[float]  # Additional context features
    
    # User state information
    user_emotional_state: Dict[str, float] = field(default_factory=dict)
    time_features: Dict[str, float] = field(default_factory=dict)
    memory_stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_vector(self) -> List[float]:
        """Convert state to flat vector for neural network input"""
        vector = []
        
        # Add feature vector (90 dims)
        vector.extend(self.feature_vector[:90])
        
        # Add context vector (variable size, pad/truncate to 20)
        context = self.context_vector[:20] + [0.0] * (20 - len(self.context_vector[:20]))
        vector.extend(context)
        
        # Add emotional state features (8 dims)
        emotions = [
            self.user_emotional_state.get('joy', 0.0),
            self.user_emotional_state.get('sadness', 0.0),
            self.user_emotional_state.get('anger', 0.0),
            self.user_emotional_state.get('fear', 0.0),
            self.user_emotional_state.get('surprise', 0.0),
            self.user_emotional_state.get('disgust', 0.0),
            self.user_emotional_state.get('trust', 0.0),
            self.user_emotional_state.get('anticipation', 0.0)
        ]
        vector.extend(emotions)
        
        # Add time features (4 dims)
        time_feats = [
            self.time_features.get('hour_norm', 0.5),  # Normalized hour
            self.time_features.get('day_of_week_norm', 0.5),
            self.time_features.get('is_weekend', 0.0),
            self.time_features.get('is_evening', 0.0)
        ]
        vector.extend(time_feats)
        
        # Add memory statistics (8 dims)
        mem_stats = [
            self.memory_stats.get('total_memories', 0) / 1000.0,  # Normalized
            self.memory_stats.get('recent_memories', 0) / 100.0,
            self.memory_stats.get('avg_importance', 0.5),
            self.memory_stats.get('memory_diversity', 0.5),
            self.memory_stats.get('access_frequency_avg', 0) / 10.0,
            self.memory_stats.get('forget_gate_avg', 0.5),
            self.memory_stats.get('input_gate_avg', 0.5),
            self.memory_stats.get('output_gate_avg', 0.5)
        ]
        vector.extend(mem_stats)
        
        # Total: 90 + 20 + 8 + 4 + 8 = 130 dimensions
        return vector
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'user_id': self.user_id,
            'feature_vector': self.feature_vector,
            'context_vector': self.context_vector,
            'user_emotional_state': self.user_emotional_state,
            'time_features': self.time_features,
            'memory_stats': self.memory_stats
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create from dictionary"""
        return cls(
            user_id=data['user_id'],
            feature_vector=data['feature_vector'],
            context_vector=data['context_vector'],
            user_emotional_state=data.get('user_emotional_state', {}),
            time_features=data.get('time_features', {}),
            memory_stats=data.get('memory_stats', {})
        )

@dataclass
class RLAction:
    """Action representation for RL training"""
    forget_threshold_adj: float  # Adjustment to forget gate threshold (-1 to 1)
    input_threshold_adj: float   # Adjustment to input gate threshold (-1 to 1)  
    output_threshold_adj: float  # Adjustment to output gate threshold (-1 to 1)
    confidence_modifier: float   # Confidence in this action (0 to 1)
    
    def __post_init__(self):
        """Validate action values"""
        self.forget_threshold_adj = np.clip(self.forget_threshold_adj, -1.0, 1.0)
        self.input_threshold_adj = np.clip(self.input_threshold_adj, -1.0, 1.0)
        self.output_threshold_adj = np.clip(self.output_threshold_adj, -1.0, 1.0)
        self.confidence_modifier = np.clip(self.confidence_modifier, 0.0, 1.0)
    
    def to_vector(self) -> List[float]:
        """Convert action to vector for neural network"""
        return [
            self.forget_threshold_adj,
            self.input_threshold_adj,
            self.output_threshold_adj,
            self.confidence_modifier
        ]
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization"""
        return {
            'forget_threshold_adj': self.forget_threshold_adj,
            'input_threshold_adj': self.input_threshold_adj,
            'output_threshold_adj': self.output_threshold_adj,
            'confidence_modifier': self.confidence_modifier
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]):
        """Create from dictionary"""
        return cls(
            forget_threshold_adj=data['forget_threshold_adj'],
            input_threshold_adj=data['input_threshold_adj'],
            output_threshold_adj=data['output_threshold_adj'],
            confidence_modifier=data['confidence_modifier']
        )

@dataclass  
class RLReward:
    """Reward components for RL training"""
    user_engagement: float      # How engaged user was (0-1)
    context_relevance: float    # How relevant retrieved memories were (0-1)
    memory_efficiency: float    # How efficiently memory was used (0-1)
    user_satisfaction: float    # Explicit user satisfaction (0-1)
    conversation_quality: float # Quality of conversation flow (0-1)
    
    # Reward component weights
    weights: Dict[str, float] = field(default_factory=lambda: {
        'user_engagement': 0.25,
        'context_relevance': 0.30,
        'memory_efficiency': 0.20,
        'user_satisfaction': 0.15,
        'conversation_quality': 0.10
    })
    
    def __post_init__(self):
        """Validate reward components"""
        self.user_engagement = np.clip(self.user_engagement, 0.0, 1.0)
        self.context_relevance = np.clip(self.context_relevance, 0.0, 1.0)
        self.memory_efficiency = np.clip(self.memory_efficiency, 0.0, 1.0)
        self.user_satisfaction = np.clip(self.user_satisfaction, 0.0, 1.0)
        self.conversation_quality = np.clip(self.conversation_quality, 0.0, 1.0)
    
    def calculate_total_reward(self) -> float:
        """Calculate weighted total reward"""
        total_reward = (
            self.user_engagement * self.weights['user_engagement'] +
            self.context_relevance * self.weights['context_relevance'] +
            self.memory_efficiency * self.weights['memory_efficiency'] +
            self.user_satisfaction * self.weights['user_satisfaction'] +
            self.conversation_quality * self.weights['conversation_quality']
        )
        return np.clip(total_reward, 0.0, 1.0)
    
    def get_component_breakdown(self) -> Dict[str, float]:
        """Get breakdown of reward components"""
        return {
            'user_engagement': self.user_engagement * self.weights['user_engagement'],
            'context_relevance': self.context_relevance * self.weights['context_relevance'],
            'memory_efficiency': self.memory_efficiency * self.weights['memory_efficiency'],
            'user_satisfaction': self.user_satisfaction * self.weights['user_satisfaction'],
            'conversation_quality': self.conversation_quality * self.weights['conversation_quality'],
            'total': self.calculate_total_reward()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'user_engagement': self.user_engagement,
            'context_relevance': self.context_relevance,
            'memory_efficiency': self.memory_efficiency,
            'user_satisfaction': self.user_satisfaction,
            'conversation_quality': self.conversation_quality,
            'weights': self.weights
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create from dictionary"""
        return cls(
            user_engagement=data['user_engagement'],
            context_relevance=data['context_relevance'],
            memory_efficiency=data['memory_efficiency'],
            user_satisfaction=data['user_satisfaction'],
            conversation_quality=data['conversation_quality'],
            weights=data.get('weights', {})
        )

@dataclass
class RLExperience:
    """Single experience tuple for RL training"""
    id: str
    user_id: str
    episode_id: str  # Conversation session ID
    step_id: int     # Step within episode
    
    state: RLState
    action: RLAction
    reward: RLReward
    next_state: Optional[RLState]
    done: bool
    
    # Training metadata
    priority: float = 1.0  # For prioritized experience replay
    importance_weight: float = 1.0  # Importance sampling weight
    td_error: Optional[float] = None  # Temporal difference error
    
    # Timestamps - FIXED with timezone-aware datetime
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        """Initialize experience with ID if not provided"""
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def get_total_reward(self) -> float:
        """Get total reward value"""
        return self.reward.calculate_total_reward()
    
    def update_priority(self, new_td_error: float, alpha: float = 0.6):
        """Update priority based on TD error for prioritized replay"""
        self.td_error = new_td_error
        self.priority = (abs(new_td_error) + 1e-6) ** alpha
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'episode_id': self.episode_id,
            'step_id': self.step_id,
            'state': self.state.to_dict(),
            'action': self.action.to_dict(),
            'reward': self.reward.to_dict(),
            'next_state': self.next_state.to_dict() if self.next_state else None,
            'done': self.done,
            'priority': self.priority,
            'importance_weight': self.importance_weight,
            'td_error': self.td_error,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create from dictionary"""
        state = RLState.from_dict(data['state'])
        action = RLAction(
            forget_threshold_adj=data['action']['forget_threshold_adj'],
            input_threshold_adj=data['action']['input_threshold_adj'],
            output_threshold_adj=data['action']['output_threshold_adj'],
            confidence_modifier=data['action']['confidence_modifier']
        )
        reward = RLReward(
            user_engagement=data['reward']['user_engagement'],
            context_relevance=data['reward']['context_relevance'],
            memory_efficiency=data['reward']['memory_efficiency'],
            user_satisfaction=data['reward']['user_satisfaction'],
            conversation_quality=data['reward']['conversation_quality']
        )
        
        next_state = None
        if data.get('next_state'):
            next_state = RLState.from_dict(data['next_state'])
        
        # Handle datetime parsing - FIXED with timezone handling
        created_at = data.get('created_at')
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        elif isinstance(created_at, datetime) and created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)
        
        return cls(
            id=data['id'],
            user_id=data['user_id'],
            episode_id=data['episode_id'],
            step_id=data['step_id'],
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=data['done'],
            priority=data.get('priority', 1.0),
            importance_weight=data.get('importance_weight', 1.0),
            td_error=data.get('td_error'),
            created_at=created_at
        )

@dataclass
class ExperienceBatch:
    """Batch of experiences for training"""
    experiences: List[RLExperience]
    batch_weights: Optional[List[float]] = None  # For importance sampling
    
    def __post_init__(self):
        """Validate batch"""
        if not self.experiences:
            raise ValueError("Experience batch cannot be empty")
        
        if self.batch_weights and len(self.batch_weights) != len(self.experiences):
            raise ValueError("Batch weights must match number of experiences")
    
    def get_states(self) -> List[List[float]]:
        """Get state vectors for batch"""
        return [exp.state.to_vector() for exp in self.experiences]
    
    def get_actions(self) -> List[List[float]]:
        """Get action vectors for batch"""
        return [exp.action.to_vector() for exp in self.experiences]
    
    def get_rewards(self) -> List[float]:
        """Get reward values for batch"""
        return [exp.get_total_reward() for exp in self.experiences]
    
    def get_next_states(self) -> List[Optional[List[float]]]:
        """Get next state vectors for batch"""
        return [exp.next_state.to_vector() if exp.next_state else None 
                for exp in self.experiences]
    
    def get_dones(self) -> List[bool]:
        """Get done flags for batch"""
        return [exp.done for exp in self.experiences]
    
    def get_priorities(self) -> List[float]:
        """Get priorities for batch"""
        return [exp.priority for exp in self.experiences]
    
    def size(self) -> int:
        """Get batch size"""
        return len(self.experiences)

# Helper functions
def create_rl_state(user_id: str, 
                   feature_vector: List[float],
                   context_data: Dict[str, Any],
                   memory_stats: Dict[str, Any]) -> RLState:
    """Helper to create RLState from components"""
    return RLState(
        user_id=user_id,
        feature_vector=feature_vector,
        context_vector=context_data.get('context_vector', []),
        user_emotional_state=context_data.get('emotions', {}),
        time_features=context_data.get('time_features', {}),
        memory_stats=memory_stats
    )

def calculate_engagement_reward(interaction_data: Dict[str, Any]) -> float:
    """Calculate engagement reward from interaction data"""
    engagement_score = 0.0
    
    # Response length (normalized)
    response_length = interaction_data.get('response_length', 0)
    engagement_score += min(1.0, response_length / 200) * 0.3
    
    # Follow-up questions
    if interaction_data.get('has_follow_ups', False):
        engagement_score += 0.3
    
    # Session duration
    session_duration = interaction_data.get('session_duration_minutes', 0)
    engagement_score += min(1.0, session_duration / 15) * 0.2
    
    # User initiated conversation
    if interaction_data.get('user_initiated', False):
        engagement_score += 0.2
    
    return min(1.0, engagement_score)