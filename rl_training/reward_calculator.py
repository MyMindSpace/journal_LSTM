# rl_training/reward_calculator.py
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

from models.rl_experience import RLReward

logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    """Types of user feedback"""
    POSITIVE_EXPLICIT = "positive_explicit"  # User says "great", "helpful", etc.
    NEGATIVE_EXPLICIT = "negative_explicit"  # User says "wrong", "unhelpful", etc.
    ENGAGEMENT = "engagement"                # User continues conversation
    DISENGAGEMENT = "disengagement"          # User ends conversation abruptly
    CORRECTION = "correction"                # User corrects AI response
    FOLLOW_UP = "follow_up"                  # User asks follow-up questions

@dataclass
class InteractionMetrics:
    """Metrics from a single user interaction"""
    response_length: int = 0
    response_time_seconds: float = 0.0
    has_follow_up_questions: bool = False
    has_emotional_indicators: bool = False
    contains_corrections: bool = False
    session_duration_minutes: float = 0.0
    context_relevance_score: float = 0.5
    memory_usage_count: int = 0
    user_satisfaction_explicit: Optional[float] = None  # 0-1 if explicitly provided

class RewardCalculator:
    """
    Comprehensive reward calculator for LSTM gate optimization
    Combines multiple signals to create meaningful reward signals
    """
    
    def __init__(self):
        # Reward component weights (should sum to 1.0)
        self.weights = {
            'user_engagement': 0.25,
            'context_relevance': 0.25,
            'memory_efficiency': 0.20,
            'user_satisfaction': 0.20,
            'conversation_quality': 0.10
        }
        
        # Engagement scoring parameters
        self.engagement_params = {
            'response_length_target': 100,  # Optimal response length
            'session_duration_target': 10,  # Target session duration (minutes)
            'follow_up_bonus': 0.3,        # Bonus for follow-up questions
            'emotional_engagement_bonus': 0.2,  # Bonus for emotional responses
        }
        
        # Context relevance parameters
        self.relevance_params = {
            'similarity_threshold': 0.6,    # Min similarity for "relevant"
            'memory_freshness_decay': 0.05, # Per day decay for memory freshness
            'context_diversity_bonus': 0.1, # Bonus for diverse memory types
        }
        
        # Efficiency parameters
        self.efficiency_params = {
            'token_budget_target': 1500,    # Optimal token usage
            'memory_count_target': 8,       # Optimal number of memories
            'processing_time_penalty': 0.1, # Penalty per second over target
        }
        
        # Historical performance tracking
        self.user_baselines: Dict[str, Dict[str, float]] = {}
        self.global_metrics = {
            'avg_engagement': 0.5,
            'avg_relevance': 0.5,
            'avg_efficiency': 0.5,
            'avg_satisfaction': 0.5
        }
    
    def calculate_engagement_reward(self, metrics: InteractionMetrics, user_id: str = None) -> float:
        """Calculate reward based on user engagement signals"""
        engagement_score = 0.0
        
        # Response length scoring (bell curve around target)
        length_score = self._calculate_bell_curve_score(
            metrics.response_length, 
            self.engagement_params['response_length_target'],
            width=50
        )
        engagement_score += length_score * 0.3
        
        # Session duration scoring
        duration_score = self._calculate_bell_curve_score(
            metrics.session_duration_minutes,
            self.engagement_params['session_duration_target'],
            width=5
        )
        engagement_score += duration_score * 0.2
        
        # Follow-up questions bonus
        if metrics.has_follow_up_questions:
            engagement_score += self.engagement_params['follow_up_bonus']
        
        # Emotional engagement bonus
        if metrics.has_emotional_indicators:
            engagement_score += self.engagement_params['emotional_engagement_bonus']
        
        # Response time penalty (too slow = bad UX)
        if metrics.response_time_seconds > 2.0:
            time_penalty = min(0.3, (metrics.response_time_seconds - 2.0) * 0.05)
            engagement_score -= time_penalty
        
        # Correction penalty
        if metrics.contains_corrections:
            engagement_score -= 0.2
        
        # Normalize and compare to user baseline
        engagement_score = np.clip(engagement_score, 0.0, 1.0)
        
        if user_id and user_id in self.user_baselines:
            baseline = self.user_baselines[user_id].get('engagement', 0.5)
            engagement_score = 0.7 * engagement_score + 0.3 * (engagement_score - baseline + 0.5)
        
        return engagement_score
    
    def calculate_context_relevance_reward(self, 
                                         context_data: Dict[str, Any],
                                         user_feedback: Optional[Dict[str, Any]] = None) -> float:
        """Calculate reward based on context relevance"""
        relevance_score = 0.0
        
        # Base relevance from similarity scores
        if 'memory_similarities' in context_data:
            similarities = context_data['memory_similarities']
            if similarities:
                avg_similarity = np.mean(similarities)
                relevance_score += min(1.0, avg_similarity / self.relevance_params['similarity_threshold']) * 0.4
        
        # Memory freshness scoring
        if 'memory_ages_days' in context_data:
            ages = context_data['memory_ages_days']
            if ages:
                freshness_scores = [max(0.1, 1.0 - age * self.relevance_params['memory_freshness_decay']) for age in ages]
                avg_freshness = np.mean(freshness_scores)
                relevance_score += avg_freshness * 0.3
        
        # Context diversity bonus
        if 'memory_types' in context_data:
            unique_types = len(set(context_data['memory_types']))
            diversity_bonus = min(self.relevance_params['context_diversity_bonus'], unique_types * 0.025)
            relevance_score += diversity_bonus
        
        # User feedback on relevance
        if user_feedback and 'context_relevance' in user_feedback:
            explicit_relevance = user_feedback['context_relevance']
            relevance_score = 0.6 * relevance_score + 0.4 * explicit_relevance
        
        # Temporal relevance bonus
        if 'temporal_matches' in context_data:
            temporal_bonus = min(0.2, context_data['temporal_matches'] * 0.05)
            relevance_score += temporal_bonus
        
        return np.clip(relevance_score, 0.0, 1.0)
    
    def calculate_memory_efficiency_reward(self, efficiency_data: Dict[str, Any]) -> float:
        """Calculate reward based on memory system efficiency"""
        efficiency_score = 0.0
        
        # Token usage efficiency
        tokens_used = efficiency_data.get('tokens_used', 0)
        token_efficiency = self._calculate_bell_curve_score(
            tokens_used,
            self.efficiency_params['token_budget_target'],
            width=300
        )
        efficiency_score += token_efficiency * 0.4
        
        # Memory count efficiency
        memories_used = efficiency_data.get('memories_used', 0)
        memory_efficiency = self._calculate_bell_curve_score(
            memories_used,
            self.efficiency_params['memory_count_target'],
            width=3
        )
        efficiency_score += memory_efficiency * 0.3
        
        # Processing time efficiency
        processing_time = efficiency_data.get('processing_time_seconds', 0.5)
        if processing_time > 1.0:  # Target: under 1 second
            time_penalty = min(0.3, (processing_time - 1.0) * self.efficiency_params['processing_time_penalty'])
            efficiency_score -= time_penalty
        else:
            efficiency_score += 0.1  # Bonus for fast processing
        
        # Memory hit rate
        hit_rate = efficiency_data.get('cache_hit_rate', 0.5)
        efficiency_score += hit_rate * 0.2
        
        return np.clip(efficiency_score, 0.0, 1.0)
    
    def calculate_user_satisfaction_reward(self, 
                                         satisfaction_data: Dict[str, Any],
                                         user_id: str = None) -> float:
        """Calculate reward based on user satisfaction signals"""
        satisfaction_score = 0.0
        
        # Explicit satisfaction rating
        if 'explicit_rating' in satisfaction_data:
            explicit_rating = satisfaction_data['explicit_rating']  # Expect 0-1 scale
            satisfaction_score += explicit_rating * 0.6
        
        # Implicit satisfaction signals
        feedback_signals = satisfaction_data.get('feedback_signals', [])
        
        positive_signals = sum(1 for signal in feedback_signals if signal in [
            FeedbackType.POSITIVE_EXPLICIT, FeedbackType.ENGAGEMENT, FeedbackType.FOLLOW_UP
        ])
        
        negative_signals = sum(1 for signal in feedback_signals if signal in [
            FeedbackType.NEGATIVE_EXPLICIT, FeedbackType.DISENGAGEMENT, FeedbackType.CORRECTION
        ])
        
        total_signals = len(feedback_signals) if feedback_signals else 1
        implicit_satisfaction = (positive_signals - negative_signals) / total_signals
        satisfaction_score += max(0, 0.5 + implicit_satisfaction * 0.5) * 0.4
        
        # Conversation continuation bonus
        if satisfaction_data.get('conversation_continued', False):
            satisfaction_score += 0.1
        
        # User return rate (long-term satisfaction)
        if user_id and 'return_rate' in satisfaction_data:
            return_rate = satisfaction_data['return_rate']
            satisfaction_score += return_rate * 0.1
        
        return np.clip(satisfaction_score, 0.0, 1.0)
    
    def calculate_conversation_quality_reward(self, quality_data: Dict[str, Any]) -> float:
        """Calculate reward based on overall conversation quality"""
        quality_score = 0.0
        
        # Conversation flow score
        flow_score = quality_data.get('flow_score', 0.5)  # How natural the conversation feels
        quality_score += flow_score * 0.4
        
        # Coherence score
        coherence_score = quality_data.get('coherence_score', 0.5)  # How well responses relate to context
        quality_score += coherence_score * 0.3
        
        # Informativeness score
        informativeness = quality_data.get('informativeness', 0.5)  # How informative responses are
        quality_score += informativeness * 0.2
        
        # Emotional appropriateness
        emotional_appropriateness = quality_data.get('emotional_appropriateness', 0.5)
        quality_score += emotional_appropriateness * 0.1
        
        return np.clip(quality_score, 0.0, 1.0)
    
    def calculate_comprehensive_reward(self,
                                     interaction_metrics: InteractionMetrics,
                                     context_data: Dict[str, Any],
                                     efficiency_data: Dict[str, Any],
                                     satisfaction_data: Dict[str, Any],
                                     quality_data: Dict[str, Any],
                                     user_id: str = None) -> RLReward:
        """Calculate comprehensive reward from all components"""
        
        # Calculate individual reward components
        engagement = self.calculate_engagement_reward(interaction_metrics, user_id)
        relevance = self.calculate_context_relevance_reward(context_data, satisfaction_data)
        efficiency = self.calculate_memory_efficiency_reward(efficiency_data)
        satisfaction = self.calculate_user_satisfaction_reward(satisfaction_data, user_id)
        quality = self.calculate_conversation_quality_reward(quality_data)
        
        # Create comprehensive reward
        reward = RLReward(
            user_engagement=engagement,
            context_relevance=relevance,
            memory_efficiency=efficiency,
            user_satisfaction=satisfaction,
            conversation_quality=quality,
            weights=self.weights.copy()
        )
        
        # Update user baseline if provided
        if user_id:
            self._update_user_baseline(user_id, {
                'engagement': engagement,
                'relevance': relevance,
                'efficiency': efficiency,
                'satisfaction': satisfaction,
                'quality': quality
            })
        
        return reward
    
    def calculate_simple_reward(self, feedback_data: Dict[str, Any]) -> float:
        """Calculate simplified reward for quick feedback"""
        # For cases where we only have basic feedback data
        engagement = feedback_data.get('user_engagement', 0.5)
        relevance = feedback_data.get('context_relevance', 0.5)
        efficiency = feedback_data.get('memory_efficiency', 0.5)
        satisfaction = feedback_data.get('user_satisfaction', 0.5)
        quality = feedback_data.get('conversation_quality', 0.5)
        
        total_reward = (
            engagement * self.weights['user_engagement'] +
            relevance * self.weights['context_relevance'] +
            efficiency * self.weights['memory_efficiency'] +
            satisfaction * self.weights['user_satisfaction'] +
            quality * self.weights['conversation_quality']
        )
        
        return np.clip(total_reward, 0.0, 1.0)
    
    def _calculate_bell_curve_score(self, value: float, target: float, width: float) -> float:
        """Calculate score using bell curve around target value"""
        if width == 0:
            return 1.0 if value == target else 0.0
        
        deviation = abs(value - target) / width
        score = np.exp(-0.5 * deviation ** 2)  # Gaussian-like curve
        return score
    
    def _update_user_baseline(self, user_id: str, scores: Dict[str, float]):
        """Update user baseline scores for personalized rewards"""
        if user_id not in self.user_baselines:
            self.user_baselines[user_id] = scores.copy()
        else:
            # Exponential moving average
            alpha = 0.1
            for metric, score in scores.items():
                if metric in self.user_baselines[user_id]:
                    self.user_baselines[user_id][metric] = (
                        alpha * score + (1 - alpha) * self.user_baselines[user_id][metric]
                    )
                else:
                    self.user_baselines[user_id][metric] = score
    
    def get_reward_breakdown(self, reward: RLReward) -> Dict[str, Dict[str, float]]:
        """Get detailed breakdown of reward components"""
        breakdown = reward.get_component_breakdown()
        
        return {
            'components': {
                'user_engagement': {
                    'raw_score': reward.user_engagement,
                    'weighted_score': breakdown['user_engagement'],
                    'weight': self.weights['user_engagement']
                },
                'context_relevance': {
                    'raw_score': reward.context_relevance,
                    'weighted_score': breakdown['context_relevance'],
                    'weight': self.weights['context_relevance']
                },
                'memory_efficiency': {
                    'raw_score': reward.memory_efficiency,
                    'weighted_score': breakdown['memory_efficiency'],
                    'weight': self.weights['memory_efficiency']
                },
                'user_satisfaction': {
                    'raw_score': reward.user_satisfaction,
                    'weighted_score': breakdown['user_satisfaction'],
                    'weight': self.weights['user_satisfaction']
                },
                'conversation_quality': {
                    'raw_score': reward.conversation_quality,
                    'weighted_score': breakdown['conversation_quality'],
                    'weight': self.weights['conversation_quality']
                }
            },
            'total': breakdown['total']
        }
    
    def analyze_reward_trends(self, user_id: str, recent_rewards: List[RLReward]) -> Dict[str, Any]:
        """Analyze reward trends for a user"""
        if not recent_rewards:
            return {}
        
        # Extract component scores
        engagement_scores = [r.user_engagement for r in recent_rewards]
        relevance_scores = [r.context_relevance for r in recent_rewards]
        efficiency_scores = [r.memory_efficiency for r in recent_rewards]
        satisfaction_scores = [r.user_satisfaction for r in recent_rewards]
        quality_scores = [r.conversation_quality for r in recent_rewards]
        total_scores = [r.calculate_total_reward() for r in recent_rewards]
        
        return {
            'user_id': user_id,
            'num_rewards': len(recent_rewards),
            'trends': {
                'engagement': {
                    'mean': np.mean(engagement_scores),
                    'std': np.std(engagement_scores),
                    'trend': self._calculate_trend(engagement_scores)
                },
                'relevance': {
                    'mean': np.mean(relevance_scores),
                    'std': np.std(relevance_scores),
                    'trend': self._calculate_trend(relevance_scores)
                },
                'efficiency': {
                    'mean': np.mean(efficiency_scores),
                    'std': np.std(efficiency_scores),
                    'trend': self._calculate_trend(efficiency_scores)
                },
                'satisfaction': {
                    'mean': np.mean(satisfaction_scores),
                    'std': np.std(satisfaction_scores),
                    'trend': self._calculate_trend(satisfaction_scores)
                },
                'quality': {
                    'mean': np.mean(quality_scores),
                    'std': np.std(quality_scores),
                    'trend': self._calculate_trend(quality_scores)
                },
                'total': {
                    'mean': np.mean(total_scores),
                    'std': np.std(total_scores),
                    'trend': self._calculate_trend(total_scores)
                }
            },
            'baseline_comparison': self.user_baselines.get(user_id, {}),
            'improvement_suggestions': self._get_improvement_suggestions(recent_rewards)
        }
    
    def _calculate_trend(self, scores: List[float]) -> str:
        """Calculate trend direction (improving, declining, stable)"""
        if len(scores) < 3:
            return "insufficient_data"
        
        # Linear regression slope
        x = np.arange(len(scores))
        slope = np.polyfit(x, scores, 1)[0]
        
        if slope > 0.05:
            return "improving"
        elif slope < -0.05:
            return "declining"
        else:
            return "stable"
    
    def _get_improvement_suggestions(self, recent_rewards: List[RLReward]) -> List[str]:
        """Get suggestions for improving rewards"""
        suggestions = []
        
        if not recent_rewards:
            return suggestions
        
        # Analyze which components are consistently low
        avg_scores = {
            'engagement': np.mean([r.user_engagement for r in recent_rewards]),
            'relevance': np.mean([r.context_relevance for r in recent_rewards]),
            'efficiency': np.mean([r.memory_efficiency for r in recent_rewards]),
            'satisfaction': np.mean([r.user_satisfaction for r in recent_rewards]),
            'quality': np.mean([r.conversation_quality for r in recent_rewards])
        }
        
        threshold = 0.6  # Below this is considered "low"
        
        if avg_scores['engagement'] < threshold:
            suggestions.append("Focus on improving user engagement through more interactive responses")
        
        if avg_scores['relevance'] < threshold:
            suggestions.append("Improve context relevance by fine-tuning memory selection")
        
        if avg_scores['efficiency'] < threshold:
            suggestions.append("Optimize memory efficiency and reduce processing time")
        
        if avg_scores['satisfaction'] < threshold:
            suggestions.append("Address user satisfaction through better response quality")
        
        if avg_scores['quality'] < threshold:
            suggestions.append("Enhance conversation quality and coherence")
        
        return suggestions