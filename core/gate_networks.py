# core/gate_networks.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from datetime import datetime,timezone

class ForgetGate(nn.Module):
    """
    Forget Gate Network - Decides what memories should fade over time
    
    Input: 90-dim feature vector (from Component 4)
    Output: Probability (0-1) that memory should be forgotten
    """
    
    def __init__(self, input_size: int = 90, hidden_size: int = 64, dropout: float = 0.1):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Output 0-1 probability
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(x)
    
    def predict_forget_probability(self, feature_vector: List[float]) -> float:
        """Predict forget probability for a single memory"""
        with torch.no_grad():
            x = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)
            prob = self.forward(x)
            return prob.item()


class InputGate(nn.Module):
    """
    Input Gate Network - Decides importance of new memories
    
    Input: 90-dim feature vector (from Component 4)
    Output: Importance score (0-1) for new memory storage
    """
    
    def __init__(self, input_size: int = 90, hidden_size: int = 64, dropout: float = 0.1):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Output 0-1 importance
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(x)
    
    def predict_importance(self, feature_vector: List[float]) -> float:
        """Predict importance score for a new memory"""
        with torch.no_grad():
            x = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)
            importance = self.forward(x)
            return importance.item()


class OutputGate(nn.Module):
    """
    Output Gate Network - Decides relevance of memories for current context
    
    Input: 90-dim feature vector + optional context features
    Output: Relevance score (0-1) for current conversation
    """
    
    def __init__(self, input_size: int = 90, context_size: int = 0, 
                 hidden_size: int = 64, dropout: float = 0.1):
        super().__init__()
        
        total_input = input_size + context_size
        
        self.network = nn.Sequential(
            nn.Linear(total_input, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Output 0-1 relevance
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with optional context"""
        if context is not None:
            x = torch.cat([x, context], dim=-1)
        return self.network(x)
    
    def predict_relevance(self, feature_vector: List[float], 
                         context_vector: Optional[List[float]] = None) -> float:
        """Predict relevance score for current context"""
        with torch.no_grad():
            x = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)
            context_tensor = None
            if context_vector:
                context_tensor = torch.tensor(context_vector, dtype=torch.float32).unsqueeze(0)
            
            relevance = self.forward(x, context_tensor)
            return relevance.item()


class LSTMGateNetwork(nn.Module):
    """
    Combined LSTM-inspired gate network system
    Manages all three gates and provides unified interface
    """
    
    def __init__(self, input_size: int = 90, hidden_size: int = 64, 
                 context_size: int = 0, dropout: float = 0.1):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        
        # Initialize the three gates
        self.forget_gate = ForgetGate(input_size, hidden_size, dropout)
        self.input_gate = InputGate(input_size, hidden_size, dropout)
        self.output_gate = OutputGate(input_size, context_size, hidden_size, dropout)
        
        # Gate thresholds (can be learned via RL)
        self.forget_threshold = nn.Parameter(torch.tensor(0.3))
        self.input_threshold = nn.Parameter(torch.tensor(0.4))
        self.output_threshold = nn.Parameter(torch.tensor(0.4))
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through all gates"""
        forget_score = self.forget_gate(x)
        input_score = self.input_gate(x)
        output_score = self.output_gate(x, context)
        
        return {
            'forget': forget_score,
            'input': input_score,
            'output': output_score
        }
    
    def get_gate_decisions(self, feature_vector: List[float], 
                          context_vector: Optional[List[float]] = None) -> Dict[str, Dict]:
        """Get gate decisions with thresholds applied"""
        with torch.no_grad():
            x = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)
            context_tensor = None
            if context_vector:
                context_tensor = torch.tensor(context_vector, dtype=torch.float32).unsqueeze(0)
            
            scores = self.forward(x, context_tensor)
            
            return {
                'forget': {
                    'score': scores['forget'].item(),
                    'decision': scores['forget'].item() > self.forget_threshold.item(),
                    'threshold': self.forget_threshold.item()
                },
                'input': {
                    'score': scores['input'].item(),
                    'decision': scores['input'].item() > self.input_threshold.item(),
                    'threshold': self.input_threshold.item()
                },
                'output': {
                    'score': scores['output'].item(),
                    'decision': scores['output'].item() > self.output_threshold.item(),
                    'threshold': self.output_threshold.item()
                }
            }
    
    def update_thresholds(self, new_thresholds: Dict[str, float]):
        """Update gate thresholds (for RL optimization)"""
        if 'forget' in new_thresholds:
            self.forget_threshold.data = torch.tensor(new_thresholds['forget'])
        if 'input' in new_thresholds:
            self.input_threshold.data = torch.tensor(new_thresholds['input'])
        if 'output' in new_thresholds:
            self.output_threshold.data = torch.tensor(new_thresholds['output'])
    
    def get_thresholds(self) -> Dict[str, float]:
        """Get current thresholds"""
        return {
            'forget': self.forget_threshold.item(),
            'input': self.input_threshold.item(),
            'output': self.output_threshold.item()
        }
    
    def save_model(self, filepath: str):
        """Save model state"""
        torch.save({
            'forget_gate_state': self.forget_gate.state_dict(),
            'input_gate_state': self.input_gate.state_dict(),
            'output_gate_state': self.output_gate.state_dict(),
            'thresholds': self.get_thresholds(),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'context_size': self.context_size
        }, filepath)
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load model from file"""
        checkpoint = torch.load(filepath)
        
        model = cls(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            context_size=checkpoint['context_size']
        )
        
        model.forget_gate.load_state_dict(checkpoint['forget_gate_state'])
        model.input_gate.load_state_dict(checkpoint['input_gate_state'])
        model.output_gate.load_state_dict(checkpoint['output_gate_state'])
        model.update_thresholds(checkpoint['thresholds'])
        
        return model


class GateEnsemble(nn.Module):
    """
    Ensemble of gate networks for improved robustness
    Uses multiple gate networks and averages their outputs
    """
    
    def __init__(self, num_models: int = 3, input_size: int = 90, 
                 hidden_size: int = 64, context_size: int = 0):
        super().__init__()
        
        self.num_models = num_models
        self.gate_networks = nn.ModuleList([
            LSTMGateNetwork(input_size, hidden_size, context_size)
            for _ in range(num_models)
        ])
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through ensemble"""
        ensemble_outputs = {'forget': [], 'input': [], 'output': []}
        
        for gate_network in self.gate_networks:
            outputs = gate_network(x, context)
            for gate_type, score in outputs.items():
                ensemble_outputs[gate_type].append(score)
        
        # Average the outputs
        averaged_outputs = {}
        for gate_type, scores in ensemble_outputs.items():
            averaged_outputs[gate_type] = torch.mean(torch.stack(scores), dim=0)
        
        return averaged_outputs
    
    def get_ensemble_decisions(self, feature_vector: List[float], 
                              context_vector: Optional[List[float]] = None) -> Dict[str, Dict]:
        """Get ensemble decisions with confidence intervals"""
        individual_decisions = []
        
        for gate_network in self.gate_networks:
            decisions = gate_network.get_gate_decisions(feature_vector, context_vector)
            individual_decisions.append(decisions)
        
        # Calculate ensemble statistics
        ensemble_results = {}
        for gate_type in ['forget', 'input', 'output']:
            scores = [d[gate_type]['score'] for d in individual_decisions]
            decisions = [d[gate_type]['decision'] for d in individual_decisions]
            
            ensemble_results[gate_type] = {
                'score_mean': np.mean(scores),
                'score_std': np.std(scores),
                'score_min': np.min(scores),
                'score_max': np.max(scores),
                'decision_consensus': np.mean(decisions),  # Fraction agreeing
                'confidence': 1.0 - np.std(scores)  # Lower std = higher confidence
            }
        
        return ensemble_results


# Helper functions for gate network utilities
def create_context_vector(query_text: str, user_state: Dict, time_features: Dict) -> List[float]:
    """Create context vector for output gate"""
    # This would typically use embeddings, but simplified for now
    context_features = []
    
    # Query length feature
    context_features.append(min(1.0, len(query_text.split()) / 50))
    
    # User emotional state features (if available)
    if 'emotions' in user_state:
        emotions = user_state['emotions']
        context_features.extend([
            emotions.get('joy', 0.0),
            emotions.get('sadness', 0.0),
            emotions.get('anger', 0.0),
            emotions.get('fear', 0.0)
        ])
    else:
        context_features.extend([0.5, 0.5, 0.5, 0.5])  # Neutral
    
    # Time features
    if time_features:
        context_features.extend([
            time_features.get('hour_normalized', 0.5),
            time_features.get('is_weekend', 0.0),
            time_features.get('is_evening', 0.0)
        ])
    else:
        context_features.extend([0.5, 0.0, 0.0])
    
    return context_features


def analyze_gate_performance(gate_decisions: List[Dict], user_feedback: List[float]) -> Dict:
    """Analyze gate performance against user feedback"""
    if len(gate_decisions) != len(user_feedback):
        raise ValueError("Decisions and feedback must have same length")
    
    # Calculate correlation between gate scores and user satisfaction
    forget_scores = [d['forget']['score'] for d in gate_decisions]
    input_scores = [d['input']['score'] for d in gate_decisions]
    output_scores = [d['output']['score'] for d in gate_decisions]
    
    forget_corr = np.corrcoef(forget_scores, user_feedback)[0, 1] if len(forget_scores) > 1 else 0.0
    input_corr = np.corrcoef(input_scores, user_feedback)[0, 1] if len(input_scores) > 1 else 0.0
    output_corr = np.corrcoef(output_scores, user_feedback)[0, 1] if len(output_scores) > 1 else 0.0
    
    return {
        'forget_gate_correlation': forget_corr,
        'input_gate_correlation': input_corr, 
        'output_gate_correlation': output_corr,
        'overall_satisfaction': np.mean(user_feedback),
        'feedback_std': np.std(user_feedback),
        'num_samples': len(user_feedback)
    }