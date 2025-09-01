# utils/token_counter.py
import re
import json
from typing import List, Dict, Optional, Union, Any
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class TokenizerType(Enum):
    """Supported tokenizer types"""
    SIMPLE = "simple"           # Simple word-based counting
    GPT = "gpt"                # GPT-style tokenization estimation
    CLAUDE = "claude"          # Claude tokenization estimation  
    BERT = "bert"              # BERT-style tokenization estimation

@dataclass
class TokenEstimate:
    """Token count estimate with metadata"""
    token_count: int
    word_count: int
    character_count: int
    tokenizer_type: str
    confidence: float  # 0-1, how confident we are in the estimate
    
    def __str__(self) -> str:
        return f"TokenEstimate(tokens={self.token_count}, words={self.word_count}, confidence={self.confidence:.2f})"

class TokenCounter:
    """
    Comprehensive token counting utility for different tokenizer types
    Provides fast estimates without requiring actual tokenizer libraries
    """
    
    def __init__(self):
        # Token-to-word ratios for different tokenizers (empirically determined)
        self.tokenizer_ratios = {
            TokenizerType.SIMPLE: 1.0,      # 1 token per word
            TokenizerType.GPT: 1.3,         # ~1.3 tokens per word
            TokenizerType.CLAUDE: 1.2,      # ~1.2 tokens per word  
            TokenizerType.BERT: 1.4,        # ~1.4 tokens per word (subword)
        }
        
        # Character-to-token ratios as fallback
        self.char_to_token_ratios = {
            TokenizerType.SIMPLE: 0.2,      # ~5 chars per token
            TokenizerType.GPT: 0.25,        # ~4 chars per token
            TokenizerType.CLAUDE: 0.23,     # ~4.3 chars per token
            TokenizerType.BERT: 0.27,       # ~3.7 chars per token
        }
        
        # Common token patterns for more accurate estimation
        self.special_patterns = {
            'numbers': re.compile(r'\b\d+\b'),
            'urls': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'emails': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'contractions': re.compile(r"\b\w+'\w+\b"),
            'punctuation_heavy': re.compile(r'[.!?;:,]{2,}'),
        }
    
    def estimate_tokens(self, 
                       text: str, 
                       tokenizer_type: TokenizerType = TokenizerType.CLAUDE,
                       method: str = 'hybrid') -> TokenEstimate:
        """
        Estimate token count for given text
        
        Args:
            text: Input text
            tokenizer_type: Type of tokenizer to estimate for
            method: Estimation method ('word_based', 'char_based', 'hybrid')
            
        Returns:
            TokenEstimate with count and confidence
        """
        if not text or not text.strip():
            return TokenEstimate(
                token_count=0,
                word_count=0,
                character_count=0,
                tokenizer_type=tokenizer_type.value,
                confidence=1.0
            )
        
        text = text.strip()
        char_count = len(text)
        word_count = self._count_words(text)
        
        if method == 'word_based':
            token_count, confidence = self._word_based_estimate(text, word_count, tokenizer_type)
        elif method == 'char_based':
            token_count, confidence = self._char_based_estimate(char_count, tokenizer_type)
        elif method == 'hybrid':
            token_count, confidence = self._hybrid_estimate(text, word_count, char_count, tokenizer_type)
        else:
            logger.warning(f"Unknown estimation method: {method}, using hybrid")
            token_count, confidence = self._hybrid_estimate(text, word_count, char_count, tokenizer_type)
        
        return TokenEstimate(
            token_count=max(1, int(token_count)),  # At least 1 token for non-empty text
            word_count=word_count,
            character_count=char_count,
            tokenizer_type=tokenizer_type.value,
            confidence=confidence
        )
    
    def _count_words(self, text: str) -> int:
        """Count words in text"""
        # Split on whitespace and filter empty strings
        words = [word for word in re.split(r'\s+', text) if word.strip()]
        return len(words)
    
    def _word_based_estimate(self, text: str, word_count: int, tokenizer_type: TokenizerType) -> tuple[float, float]:
        """Estimate tokens based on word count"""
        base_ratio = self.tokenizer_ratios[tokenizer_type]
        
        # Adjust ratio based on text characteristics
        adjustment_factor = self._calculate_adjustment_factor(text)
        adjusted_ratio = base_ratio * adjustment_factor
        
        token_count = word_count * adjusted_ratio
        
        # Confidence is higher for longer texts and standard ratios
        confidence = min(0.9, 0.6 + (word_count / 1000) * 0.3)
        
        return token_count, confidence
    
    def _char_based_estimate(self, char_count: int, tokenizer_type: TokenizerType) -> tuple[float, float]:
        """Estimate tokens based on character count"""
        ratio = self.char_to_token_ratios[tokenizer_type]
        token_count = char_count * ratio
        
        # Character-based estimates are less reliable
        confidence = 0.5
        
        return token_count, confidence
    
    def _hybrid_estimate(self, text: str, word_count: int, char_count: int, tokenizer_type: TokenizerType) -> tuple[float, float]:
        """Hybrid estimation combining word and character methods"""
        word_estimate, word_confidence = self._word_based_estimate(text, word_count, tokenizer_type)
        char_estimate, char_confidence = self._char_based_estimate(char_count, tokenizer_type)
        
        # Weight word-based estimate higher for typical text
        if word_count > 0:
            # Use word-based primarily, char-based for adjustment
            avg_word_length = char_count / word_count
            
            if avg_word_length < 3:  # Very short words, char-based might be better
                weight_word = 0.6
                weight_char = 0.4
            elif avg_word_length > 8:  # Very long words, char-based helps
                weight_word = 0.7
                weight_char = 0.3
            else:  # Normal words, word-based is best
                weight_word = 0.8
                weight_char = 0.2
            
            hybrid_estimate = (word_estimate * weight_word + char_estimate * weight_char)
            hybrid_confidence = (word_confidence * weight_word + char_confidence * weight_char)
        else:
            hybrid_estimate = char_estimate
            hybrid_confidence = char_confidence
        
        return hybrid_estimate, hybrid_confidence
    
    def _calculate_adjustment_factor(self, text: str) -> float:
        """Calculate adjustment factor based on text characteristics"""
        factor = 1.0
        
        # Numbers typically tokenize differently
        if self.special_patterns['numbers'].search(text):
            factor *= 0.9  # Numbers often tokenize as single tokens
        
        # URLs and emails
        if self.special_patterns['urls'].search(text) or self.special_patterns['emails'].search(text):
            factor *= 1.2  # URLs/emails break into many tokens
        
        # Contractions
        contractions = len(self.special_patterns['contractions'].findall(text))
        if contractions > 0:
            factor *= (1 + contractions * 0.1)  # Contractions may split
        
        # Heavy punctuation
        if self.special_patterns['punctuation_heavy'].search(text):
            factor *= 1.1  # Extra punctuation creates tokens
        
        # Code-like content (lots of symbols)
        symbol_ratio = len(re.findall(r'[{}()\[\]<>@#$%^&*+=|\\:;"]', text)) / len(text)
        if symbol_ratio > 0.1:
            factor *= 1.3  # Code tokenizes differently
        
        return max(0.5, min(2.0, factor))  # Reasonable bounds
    
    def count_tokens_in_memories(self, memory_summaries: List[str], 
                                tokenizer_type: TokenizerType = TokenizerType.CLAUDE) -> Dict[str, Any]:
        """
        Count tokens across multiple memory summaries
        
        Args:
            memory_summaries: List of memory content summaries
            tokenizer_type: Tokenizer type to estimate for
            
        Returns:
            Dictionary with detailed token statistics
        """
        if not memory_summaries:
            return {
                'total_tokens': 0,
                'individual_counts': [],
                'average_tokens_per_memory': 0,
                'token_distribution': {},
                'total_memories': 0
            }
        
        individual_estimates = []
        total_tokens = 0
        
        for i, summary in enumerate(memory_summaries):
            estimate = self.estimate_tokens(summary, tokenizer_type)
            individual_estimates.append({
                'index': i,
                'tokens': estimate.token_count,
                'words': estimate.word_count,
                'chars': estimate.character_count,
                'confidence': estimate.confidence
            })
            total_tokens += estimate.token_count
        
        # Calculate distribution
        token_counts = [est['tokens'] for est in individual_estimates]
        distribution = self._calculate_distribution(token_counts)
        
        return {
            'total_tokens': total_tokens,
            'individual_counts': individual_estimates,
            'average_tokens_per_memory': total_tokens / len(memory_summaries),
            'token_distribution': distribution,
            'total_memories': len(memory_summaries),
            'tokenizer_type': tokenizer_type.value
        }
    
    def _calculate_distribution(self, counts: List[int]) -> Dict[str, Union[int, float]]:
        """Calculate statistical distribution of token counts"""
        if not counts:
            return {}
        
        sorted_counts = sorted(counts)
        n = len(sorted_counts)
        
        return {
            'min': min(sorted_counts),
            'max': max(sorted_counts),
            'median': sorted_counts[n // 2],
            'q25': sorted_counts[n // 4],
            'q75': sorted_counts[3 * n // 4],
            'std': self._calculate_std(sorted_counts),
            'range': max(sorted_counts) - min(sorted_counts)
        }
    
    def _calculate_std(self, values: List[int]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def fit_within_budget(self, 
                         texts: List[str], 
                         max_tokens: int,
                         tokenizer_type: TokenizerType = TokenizerType.CLAUDE,
                         strategy: str = 'greedy') -> Dict[str, Any]:
        """
        Select texts that fit within token budget
        
        Args:
            texts: List of text strings
            max_tokens: Maximum token budget
            tokenizer_type: Tokenizer type to estimate for
            strategy: Selection strategy ('greedy', 'balanced', 'priority')
            
        Returns:
            Dictionary with selected texts and metadata
        """
        if not texts or max_tokens <= 0:
            return {
                'selected_texts': [],
                'selected_indices': [],
                'total_tokens_used': 0,
                'texts_considered': 0,
                'texts_selected': 0,
                'budget_utilization': 0.0
            }
        
        # Estimate tokens for each text
        estimates = []
        for i, text in enumerate(texts):
            estimate = self.estimate_tokens(text, tokenizer_type)
            estimates.append({
                'index': i,
                'text': text,
                'tokens': estimate.token_count,
                'words': estimate.word_count,
                'confidence': estimate.confidence
            })
        
        # Apply selection strategy
        if strategy == 'greedy':
            selected = self._greedy_selection(estimates, max_tokens)
        elif strategy == 'balanced':
            selected = self._balanced_selection(estimates, max_tokens)
        else:
            selected = self._greedy_selection(estimates, max_tokens)
        
        total_tokens = sum(est['tokens'] for est in selected)
        
        return {
            'selected_texts': [est['text'] for est in selected],
            'selected_indices': [est['index'] for est in selected],
            'selected_estimates': selected,
            'total_tokens_used': total_tokens,
            'texts_considered': len(texts),
            'texts_selected': len(selected),
            'budget_utilization': total_tokens / max_tokens if max_tokens > 0 else 0.0,
            'average_confidence': sum(est['confidence'] for est in selected) / len(selected) if selected else 0.0
        }
    
    def _greedy_selection(self, estimates: List[Dict], max_tokens: int) -> List[Dict]:
        """Greedy selection: take items in order until budget exhausted"""
        selected = []
        total_tokens = 0
        
        for estimate in estimates:
            if total_tokens + estimate['tokens'] <= max_tokens:
                selected.append(estimate)
                total_tokens += estimate['tokens']
            else:
                break
        
        return selected
    
    def _balanced_selection(self, estimates: List[Dict], max_tokens: int) -> List[Dict]:
        """Balanced selection: prefer smaller items for better diversity"""
        # Sort by token count (smallest first)
        sorted_estimates = sorted(estimates, key=lambda x: x['tokens'])
        return self._greedy_selection(sorted_estimates, max_tokens)

# Convenience functions for quick usage
def estimate_tokens(text: str, tokenizer: str = 'claude') -> int:
    """
    Quick token estimation function
    
    Args:
        text: Input text
        tokenizer: Tokenizer type ('gpt', 'claude', 'bert', 'simple')
        
    Returns:
        Estimated token count
    """
    counter = TokenCounter()
    
    tokenizer_map = {
        'gpt': TokenizerType.GPT,
        'claude': TokenizerType.CLAUDE,
        'bert': TokenizerType.BERT,
        'simple': TokenizerType.SIMPLE
    }
    
    tokenizer_type = tokenizer_map.get(tokenizer.lower(), TokenizerType.CLAUDE)
    estimate = counter.estimate_tokens(text, tokenizer_type)
    
    return estimate.token_count

def count_memory_tokens(memory_summaries: List[str]) -> int:
    """
    Quick function to count total tokens in memory summaries
    
    Args:
        memory_summaries: List of memory content summaries
        
    Returns:
        Total estimated token count
    """
    counter = TokenCounter()
    stats = counter.count_tokens_in_memories(memory_summaries)
    return stats['total_tokens']

def fit_memories_in_budget(memory_summaries: List[str], max_tokens: int) -> List[str]:
    """
    Select memory summaries that fit within token budget
    
    Args:
        memory_summaries: List of memory summaries
        max_tokens: Maximum token budget
        
    Returns:
        List of selected memory summaries
    """
    counter = TokenCounter()
    result = counter.fit_within_budget(memory_summaries, max_tokens)
    return result['selected_texts']

class TokenBudgetManager:
    """
    Advanced token budget management for conversation contexts
    """
    
    def __init__(self, max_tokens: int = 2000, reserve_tokens: int = 200):
        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens  # Reserved for system prompts, etc.
        self.available_tokens = max_tokens - reserve_tokens
        self.counter = TokenCounter()
        
        # Track usage
        self.current_usage = 0
        self.allocations = {}
    
    def allocate_tokens(self, component: str, tokens: int) -> bool:
        """
        Allocate tokens for a specific component
        
        Args:
            component: Component name (e.g., 'memories', 'query', 'context')
            tokens: Number of tokens to allocate
            
        Returns:
            True if allocation successful, False if insufficient tokens
        """
        if self.current_usage + tokens > self.available_tokens:
            return False
        
        self.allocations[component] = self.allocations.get(component, 0) + tokens
        self.current_usage += tokens
        return True
    
    def deallocate_tokens(self, component: str, tokens: int = None):
        """Deallocate tokens from a component"""
        if component not in self.allocations:
            return
        
        if tokens is None:
            tokens = self.allocations[component]
        
        actual_deallocation = min(tokens, self.allocations[component])
        self.allocations[component] -= actual_deallocation
        self.current_usage -= actual_deallocation
        
        if self.allocations[component] <= 0:
            del self.allocations[component]
    
    def get_remaining_tokens(self) -> int:
        """Get remaining available tokens"""
        return self.available_tokens - self.current_usage
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get detailed usage statistics"""
        return {
            'total_budget': self.max_tokens,
            'reserved_tokens': self.reserve_tokens,
            'available_tokens': self.available_tokens,
            'current_usage': self.current_usage,
            'remaining_tokens': self.get_remaining_tokens(),
            'utilization_rate': self.current_usage / self.available_tokens,
            'allocations': self.allocations.copy()
        }
    
    def reset_budget(self):
        """Reset the token budget"""
        self.current_usage = 0
        self.allocations.clear()