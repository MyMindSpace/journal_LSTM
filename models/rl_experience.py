# models/rl_experience.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
import uuid
import json
import numpy as np

@dataclass
class RLState:
    """RL State representation for gate optimization"""
    user_id: str
    feature_vector: List[float]  # 90-dim features from Component 4
    context_vector: List[float]  # Current conversation context
    user_emotional_state: Dict[str, float]  # Current emotions
    time_features: Dict[str, Any]  # Time-based features
    memory_stats: Dict[str, Any]  # User's memory statistics
    
    def to_vector(self) -> List[float]:
        """Convert state to flat vector for neural network input"""
        state_vector = []
        
        # Add feature vector (90 dims)
        state_vector.extend(self.feature_vector)
        
        # Add context vector (variable length, pad/truncate to 20)
        context_padded = self.context_vector[:20] + [0.0] * max(0, 20 - len(self.context_vector))
        state_vector.extend(context_padded)
        
        # Add emotional state (8 emotions)
        emotions = [
            self.user_emotional_state.get('joy', 0.5),
            self.user_emotional_state.get('sadness', 0.5),
            self.user_emotional_state.get('anger', 0.5),
            self.user_emotional_state.get('fear', 0.5),
            self.user_emotional_state.get('surprise', 0.5),
            self.user_emotional_state.get('disgust', 0.5),
            self.user_emotional_state.get('anticipation', 0.5),
            self.user_emotional_state.get('trust', 0.5)
        ]
        state_vector.extend(emotions)
        
        # Add time features (5 dims)
        time_vector = [
            self.time_features.get('hour_normalized', 0.5),
            self.time_features.get('day_of_week_normalized', 0.5),
            float(self.time_features.get('is_weekend', False)),
            float(self.time_features.get('is_evening', False)),
            float(self.time_features.get('is_work_hours', True))
        ]
        state_vector.extend(time_vector)
        
        # Add memory stats (7 dims)
        memory_vector = [
            min(1.0, self.memory_stats.get('total_memories', 0) / 1000),  # Normalize
            self.memory_stats.get('avg_importance', 0.5),
            self.memory_stats.get('recent_access_rate', 0.5),
            min(1.0, self.memory_stats.get('days_since_last_entry', 0) / 30),
            self.memory_stats.get('conversation_ratio', 0.5),
            self.memory_stats.get('event_ratio', 0.25),
            self.memory_stats.get('emotion_ratio', 0.25)
        ]
        state_vector.extend(memory_vector)
        
        return state_vector
    
    def get_vector_size(self) -> int:
        """Get the size of the state vector"""
        return 90 + 20 + 8 + 5 + 7  # 130 total dimensions
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create RLState from dictionary"""
        return cls(
            user_id=data['user_id'],
            feature_vector=data['feature_vector'],
            context_vector=data.get('context_vector', []),
            user_emotional_state=data.get('user_emotional_state', {}),
            time_features=data.get('time_features', {}),
            memory_stats=data.get('memory_stats', {})
        )

@dataclass
class RLAction:
    """RL Action for gate threshold adjustments"""
    forget_threshold_adj: float  # Adjustment to forget gate threshold (-0.2 to +0.2)
    input_threshold_adj: float   # Adjustment to input gate threshold (-0.2 to +0.2)
    output_threshold_adj: float  # Adjustment to output gate threshold (-0.2 to +0.2)
    confidence_modifier: float   # Confidence in this action (-0.1 to +0.1)
    
    def __post_init__(self):
        """Validate action values"""
        self.forget_threshold_adj = np.clip(self.forget_threshold_adj, -0.2, 0.2)
        self.input_threshold_adj = np.clip(self.input_threshold_adj, -0.2, 0.2)
        self.output_threshold_adj = np.clip(self.output_threshold_adj, -0.2, 0.2)
        self.confidence_modifier = np.clip(self.confidence_modifier, -0.1, 0.1)
    
    def to_vector(self) -> List[float]:
        """Convert action to vector"""
        return [
            self.forget_threshold_adj,
            self.input_threshold_adj,
            self.output_threshold_adj,
            self.confidence_modifier
        ]
    
    def apply_to_thresholds(self, current_thresholds: Dict[str, float]) -> Dict[str, float]:
        """Apply action adjustments to current thresholds"""
        new_thresholds = {
            'forget': np.clip(current_thresholds['forget'] + self.forget_threshold_adj, 0.1, 0.9),
            'input': np.clip(current_thresholds['input'] + self.input_threshold_adj, 0.1, 0.9),
            'output': np.clip(current_thresholds['output'] + self.output_threshold_adj, 0.1, 0.9)
        }
        return new_thresholds
    
    @classmethod
    def from_vector(cls, action_vector: List[float]):
        """Create action from vector"""
        if len(action_vector) != 4:
            raise ValueError("Action vector must have exactly 4 elements")
        
        return cls(
            forget_threshold_adj=action_vector[0],
            input_threshold_adj=action_vector[1],
            output_threshold_adj=action_vector[2],
            confidence_modifier=action_vector[3]
        )
    
    @classmethod
    def random_action(cls, scale: float = 0.1):
        """Generate random action for exploration"""
        return cls(
            forget_threshold_adj=np.random.uniform(-scale, scale),
            input_threshold_adj=np.random.uniform(-scale, scale),
            output_threshold_adj=np.random.uniform(-scale, scale),
            confidence_modifier=np.random.uniform(-scale/2, scale/2)
        )

@dataclass
class RLReward:
    """Reward calculation components"""
    user_engagement: float      # 0-1 score based on response length, follow-ups
    context_relevance: float    # 0-1 score for how relevant context was
    memory_efficiency: float    # 0-1 score for memory usage efficiency
    user_satisfaction: float    # 0-1 explicit user feedback
    conversation_quality: float # 0-1 overall conversation flow
    
    # Reward weights (should sum to 1.0)
    weights: Dict[str, float] = field(default_factory=lambda: {
        'user_engagement': 0.25,
        'context_relevance': 0.25,
        'memory_efficiency': 0.20,
        'user_satisfaction': 0.20,
        'conversation_quality': 0.10
    })
    
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
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    
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
        """Convert to dictionary for storage"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'episode_id': self.episode_id,
            'step_id': self.step_id,
            'state': {
                'user_id': self.state.user_id,
                'feature_vector': self.state.feature_vector,
                'context_vector': self.state.context_vector,
                'user_emotional_state': self.state.user_emotional_state,
                'time_features': self.state.time_features,
                'memory_stats': self.state.memory_stats
            },
            'action': {
                'forget_threshold_adj': self.action.forget_threshold_adj,
                'input_threshold_adj': self.action.input_threshold_adj,
                'output_threshold_adj': self.action.output_threshold_adj,
                'confidence_modifier': self.action.confidence_modifier
            },
            'reward': {
                'user_engagement': self.reward.user_engagement,
                'context_relevance': self.reward.context_relevance,
                'memory_efficiency': self.reward.memory_efficiency,
                'user_satisfaction': self.reward.user_satisfaction,
                'conversation_quality': self.reward.conversation_quality,
                'total': self.reward.calculate_total_reward()
            },
            'next_state': {
                'user_id': self.next_state.user_id,
                'feature_vector': self.next_state.feature_vector,
                'context_vector': self.next_state.context_vector,
                'user_emotional_state': self.next_state.user_emotional_state,
                'time_features': self.next_state.time_features,
                'memory_stats': self.next_state.memory_stats
            } if self.next_state else None,
            'done': self.done,
            'priority': self.priority,
            'importance_weight': self.importance_weight,
            'td_error': self.td_error,
            'created_at': self.created_at.isoformat()
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
            created_at=datetime.fromisoformat(data['created_at'])
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