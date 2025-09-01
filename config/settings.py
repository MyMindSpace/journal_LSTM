# config/settings.py
from dataclasses import dataclass
from typing import Dict, Any
import os

@dataclass
class AstraDBConfig:
    """Astra DB configuration"""
    endpoint: str
    token: str
    keyspace: str = "memory_db"
    timeout: int = 30
    
    @classmethod
    def from_env(cls):
        return cls(
            endpoint=os.getenv("ASTRA_DB_ENDPOINT", ""),
            token=os.getenv("ASTRA_DB_TOKEN", ""),
            keyspace=os.getenv("ASTRA_KEYSPACE", "memory_db")
        )

@dataclass
class GateNetworkConfig:
    """Gate network configuration"""
    input_size: int = 90  # Feature vector size from Component 4
    hidden_size: int = 64
    output_size: int = 1
    dropout_rate: float = 0.1
    activation: str = "relu"

@dataclass
class MemoryConfig:
    """Memory management configuration"""
    # Gate thresholds
    forget_threshold: float = 0.3
    input_threshold: float = 0.7
    output_threshold: float = 0.5
    
    # Context limits
    max_context_tokens: int = 2000
    max_memories_per_context: int = 20
    
    # Decay parameters
    time_decay_rate: float = 0.01  # Per day
    access_decay_rate: float = 0.02  # Per day since last access
    
    # Memory types
    memory_types: list = None
    
    def __post_init__(self):
        if self.memory_types is None:
            self.memory_types = ["conversation", "event", "emotion", "insight"]

@dataclass
class RLConfig:
    """Reinforcement learning configuration"""
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    buffer_size: int = 10000
    gamma: float = 0.99  # Discount factor
    
    # Exploration
    epsilon_start: float = 0.1
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # Training schedule
    training_interval_hours: int = 24
    min_experiences_for_training: int = 100
    training_steps_per_session: int = 10
    
    # Reward weights
    reward_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.reward_weights is None:
            self.reward_weights = {
                'response_length': 0.2,
                'follow_up_questions': 0.3,
                'positive_feedback': 0.4,
                'session_duration': 0.1,
                'context_relevance_penalty': -0.5
            }

class LSTMConfig:
    """Main configuration class"""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        if config_dict is None:
            config_dict = {}
        
        self.astra_db = AstraDBConfig.from_env()
        self.gate_network = GateNetworkConfig(**config_dict.get("gate_network", {}))
        self.memory = MemoryConfig(**config_dict.get("memory", {}))
        self.rl = RLConfig(**config_dict.get("rl", {}))
    
    def validate(self) -> bool:
        """Validate configuration"""
        if not self.astra_db.endpoint or not self.astra_db.token:
            raise ValueError("Astra DB endpoint and token must be provided")
        
        if self.memory.forget_threshold >= 1.0 or self.memory.forget_threshold <= 0.0:
            raise ValueError("Forget threshold must be between 0 and 1")
        
        if self.rl.learning_rate <= 0 or self.rl.learning_rate >= 1:
            raise ValueError("Learning rate must be between 0 and 1")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "astra_db": {
                "endpoint": self.astra_db.endpoint,
                "keyspace": self.astra_db.keyspace,
                "timeout": self.astra_db.timeout
            },
            "gate_network": {
                "input_size": self.gate_network.input_size,
                "hidden_size": self.gate_network.hidden_size,
                "dropout_rate": self.gate_network.dropout_rate
            },
            "memory": {
                "forget_threshold": self.memory.forget_threshold,
                "input_threshold": self.memory.input_threshold,
                "output_threshold": self.memory.output_threshold,
                "max_context_tokens": self.memory.max_context_tokens
            },
            "rl": {
                "learning_rate": self.rl.learning_rate,
                "batch_size": self.rl.batch_size,
                "epsilon_start": self.rl.epsilon_start,
                "training_interval_hours": self.rl.training_interval_hours
            }
        }

# Default configuration instance
DEFAULT_CONFIG = LSTMConfig()