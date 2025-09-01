# utils/similarity.py
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import math
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logger = logging.getLogger(__name__)

def cosine_similarity(vector1: List[float], vector2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors
    
    Args:
        vector1: First vector
        vector2: Second vector
        
    Returns:
        Cosine similarity score between -1 and 1
    """
    if not vector1 or not vector2:
        return 0.0
    
    if len(vector1) != len(vector2):
        logger.warning(f"Vector length mismatch: {len(vector1)} vs {len(vector2)}")
        return 0.0
    
    try:
        # Convert to numpy arrays
        v1 = np.array(vector1, dtype=np.float32)
        v2 = np.array(vector2, dtype=np.float32)
        
        # Handle zero vectors
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(v1, v2) / (norm1 * norm2)
        
        # Ensure result is in valid range
        return float(np.clip(similarity, -1.0, 1.0))
        
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {e}")
        return 0.0

def euclidean_distance(vector1: List[float], vector2: List[float]) -> float:
    """
    Calculate Euclidean distance between two vectors
    
    Args:
        vector1: First vector
        vector2: Second vector
        
    Returns:
        Euclidean distance (0 = identical, higher = more different)
    """
    if not vector1 or not vector2:
        return float('inf')
    
    if len(vector1) != len(vector2):
        logger.warning(f"Vector length mismatch: {len(vector1)} vs {len(vector2)}")
        return float('inf')
    
    try:
        v1 = np.array(vector1, dtype=np.float32)
        v2 = np.array(vector2, dtype=np.float32)
        
        distance = np.linalg.norm(v1 - v2)
        return float(distance)
        
    except Exception as e:
        logger.error(f"Error calculating Euclidean distance: {e}")
        return float('inf')

def manhattan_distance(vector1: List[float], vector2: List[float]) -> float:
    """
    Calculate Manhattan (L1) distance between two vectors
    
    Args:
        vector1: First vector
        vector2: Second vector
        
    Returns:
        Manhattan distance
    """
    if not vector1 or not vector2:
        return float('inf')
    
    if len(vector1) != len(vector2):
        return float('inf')
    
    try:
        v1 = np.array(vector1, dtype=np.float32)
        v2 = np.array(vector2, dtype=np.float32)
        
        distance = np.sum(np.abs(v1 - v2))
        return float(distance)
        
    except Exception as e:
        logger.error(f"Error calculating Manhattan distance: {e}")
        return float('inf')

def jaccard_similarity(set1: set, set2: set) -> float:
    """
    Calculate Jaccard similarity between two sets
    
    Args:
        set1: First set
        set2: Second set
        
    Returns:
        Jaccard similarity score between 0 and 1
    """
    if not set1 and not set2:
        return 1.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 0.0
    
    return intersection / union

def weighted_cosine_similarity(vector1: List[float], 
                              vector2: List[float], 
                              weights: List[float]) -> float:
    """
    Calculate weighted cosine similarity between two vectors
    
    Args:
        vector1: First vector
        vector2: Second vector
        weights: Weight for each dimension
        
    Returns:
        Weighted cosine similarity score
    """
    if len(vector1) != len(vector2) or len(vector1) != len(weights):
        logger.warning("Vector and weight length mismatch")
        return 0.0
    
    try:
        v1 = np.array(vector1, dtype=np.float32)
        v2 = np.array(vector2, dtype=np.float32)
        w = np.array(weights, dtype=np.float32)
        
        # Apply weights
        weighted_v1 = v1 * w
        weighted_v2 = v2 * w
        
        # Calculate weighted cosine similarity
        return cosine_similarity(weighted_v1.tolist(), weighted_v2.tolist())
        
    except Exception as e:
        logger.error(f"Error calculating weighted cosine similarity: {e}")
        return 0.0

def semantic_similarity(text1: str, text2: str, method: str = 'tfidf') -> float:
    """
    Calculate semantic similarity between two text strings
    
    Args:
        text1: First text
        text2: Second text
        method: Similarity method ('tfidf', 'jaccard')
        
    Returns:
        Semantic similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0
    
    if method == 'tfidf':
        return _tfidf_similarity(text1, text2)
    elif method == 'jaccard':
        return _jaccard_text_similarity(text1, text2)
    else:
        logger.warning(f"Unknown similarity method: {method}")
        return 0.0

def _tfidf_similarity(text1: str, text2: str) -> float:
    """Calculate TF-IDF based similarity"""
    try:
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        
        # Calculate cosine similarity
        similarity_matrix = sklearn_cosine(tfidf_matrix)
        return float(similarity_matrix[0, 1])
        
    except Exception as e:
        logger.error(f"Error calculating TF-IDF similarity: {e}")
        return 0.0

def _jaccard_text_similarity(text1: str, text2: str) -> float:
    """Calculate Jaccard similarity for text"""
    # Simple word-based Jaccard similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    return jaccard_similarity(words1, words2)

def batch_cosine_similarity(query_vector: List[float], 
                           vectors: List[List[float]]) -> List[float]:
    """
    Calculate cosine similarity between a query vector and multiple vectors
    
    Args:
        query_vector: Query vector
        vectors: List of vectors to compare against
        
    Returns:
        List of similarity scores
    """
    if not query_vector or not vectors:
        return []
    
    similarities = []
    for vector in vectors:
        sim = cosine_similarity(query_vector, vector)
        similarities.append(sim)
    
    return similarities

def find_most_similar(query_vector: List[float], 
                     vectors: List[List[float]], 
                     top_k: int = 5) -> List[Tuple[int, float]]:
    """
    Find the most similar vectors to a query vector
    
    Args:
        query_vector: Query vector
        vectors: List of vectors to search
        top_k: Number of top results to return
        
    Returns:
        List of (index, similarity_score) tuples, sorted by similarity
    """
    similarities = batch_cosine_similarity(query_vector, vectors)
    
    # Create (index, similarity) pairs and sort by similarity
    indexed_similarities = [(i, sim) for i, sim in enumerate(similarities)]
    indexed_similarities.sort(key=lambda x: x[1], reverse=True)
    
    return indexed_similarities[:top_k]

def similarity_matrix(vectors: List[List[float]]) -> np.ndarray:
    """
    Create a similarity matrix for a list of vectors
    
    Args:
        vectors: List of vectors
        
    Returns:
        NxN similarity matrix where N is the number of vectors
    """
    if not vectors:
        return np.array([])
    
    n = len(vectors)
    matrix = np.zeros((n, n), dtype=np.float32)
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                matrix[i, j] = 1.0
            else:
                sim = cosine_similarity(vectors[i], vectors[j])
                matrix[i, j] = sim
                matrix[j, i] = sim  # Symmetric matrix
    
    return matrix

def adaptive_similarity(vector1: List[float], 
                       vector2: List[float], 
                       context: Dict[str, any] = None) -> float:
    """
    Calculate adaptive similarity that considers context
    
    Args:
        vector1: First vector
        vector2: Second vector
        context: Additional context for similarity calculation
        
    Returns:
        Adaptive similarity score
    """
    base_similarity = cosine_similarity(vector1, vector2)
    
    if not context:
        return base_similarity
    
    # Apply contextual adjustments
    adjusted_similarity = base_similarity
    
    # Temporal context adjustment
    if 'time_weight' in context:
        time_weight = context['time_weight']
        adjusted_similarity *= time_weight
    
    # Importance weighting
    if 'importance_weight' in context:
        importance_weight = context['importance_weight']
        adjusted_similarity = (adjusted_similarity * 0.8 + 
                              importance_weight * 0.2)
    
    # User preference adjustment
    if 'user_preference_weight' in context:
        pref_weight = context['user_preference_weight']
        adjusted_similarity = (adjusted_similarity * 0.7 + 
                              pref_weight * 0.3)
    
    return np.clip(adjusted_similarity, 0.0, 1.0)

def multi_dimensional_similarity(vectors1: List[List[float]], 
                                vectors2: List[List[float]],
                                weights: Optional[List[float]] = None) -> float:
    """
    Calculate similarity across multiple dimensions/aspects
    
    Args:
        vectors1: List of vectors for first item (different aspects)
        vectors2: List of vectors for second item (different aspects)  
        weights: Optional weights for each dimension
        
    Returns:
        Multi-dimensional similarity score
    """
    if len(vectors1) != len(vectors2):
        logger.warning("Mismatch in number of dimensions")
        return 0.0
    
    if not weights:
        weights = [1.0 / len(vectors1)] * len(vectors1)
    
    if len(weights) != len(vectors1):
        logger.warning("Weights length mismatch")
        return 0.0
    
    total_similarity = 0.0
    
    for i, (v1, v2, weight) in enumerate(zip(vectors1, vectors2, weights)):
        dim_similarity = cosine_similarity(v1, v2)
        total_similarity += dim_similarity * weight
    
    return total_similarity

def fuzzy_similarity(vector1: List[float], 
                    vector2: List[float], 
                    threshold: float = 0.1) -> float:
    """
    Calculate fuzzy similarity that treats small differences as identical
    
    Args:
        vector1: First vector
        vector2: Second vector
        threshold: Threshold below which differences are ignored
        
    Returns:
        Fuzzy similarity score
    """
    if len(vector1) != len(vector2):
        return 0.0
    
    try:
        v1 = np.array(vector1, dtype=np.float32)
        v2 = np.array(vector2, dtype=np.float32)
        
        # Calculate element-wise differences
        differences = np.abs(v1 - v2)
        
        # Apply fuzzy threshold
        fuzzy_differences = np.where(differences <= threshold, 0.0, differences)
        
        # Create fuzzy vectors
        fuzzy_v1 = v1.copy()
        fuzzy_v2 = v2.copy()
        
        # Set small differences to identical values
        mask = differences <= threshold
        fuzzy_v2[mask] = fuzzy_v1[mask]
        
        return cosine_similarity(fuzzy_v1.tolist(), fuzzy_v2.tolist())
        
    except Exception as e:
        logger.error(f"Error calculating fuzzy similarity: {e}")
        return 0.0

def temporal_similarity(vector1: List[float], 
                       vector2: List[float],
                       timestamp1: float,
                       timestamp2: float,
                       time_decay_rate: float = 0.1) -> float:
    """
    Calculate similarity with temporal decay
    
    Args:
        vector1: First vector
        vector2: Second vector
        timestamp1: Timestamp of first vector (unix timestamp)
        timestamp2: Timestamp of second vector (unix timestamp)
        time_decay_rate: Rate at which similarity decays over time
        
    Returns:
        Temporal similarity score
    """
    base_similarity = cosine_similarity(vector1, vector2)
    
    # Calculate time difference in days
    time_diff_days = abs(timestamp2 - timestamp1) / (24 * 3600)
    
    # Apply temporal decay
    time_weight = math.exp(-time_decay_rate * time_diff_days)
    
    return base_similarity * time_weight

class SimilarityCalculator:
    """
    Advanced similarity calculator with caching and optimization
    """
    
    def __init__(self, cache_size: int = 1000):
        self.cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
    
    def calculate(self, 
                 vector1: List[float], 
                 vector2: List[float],
                 method: str = 'cosine',
                 use_cache: bool = True) -> float:
        """
        Calculate similarity with caching
        
        Args:
            vector1: First vector
            vector2: Second vector
            method: Similarity method ('cosine', 'euclidean', 'manhattan')
            use_cache: Whether to use caching
            
        Returns:
            Similarity score
        """
        if use_cache:
            # Create cache key
            cache_key = self._create_cache_key(vector1, vector2, method)
            
            if cache_key in self.cache:
                self.cache_hits += 1
                return self.cache[cache_key]
        
        # Calculate similarity
        if method == 'cosine':
            similarity = cosine_similarity(vector1, vector2)
        elif method == 'euclidean':
            distance = euclidean_distance(vector1, vector2)
            # Convert distance to similarity (higher = more similar)
            similarity = 1.0 / (1.0 + distance) if distance != float('inf') else 0.0
        elif method == 'manhattan':
            distance = manhattan_distance(vector1, vector2)
            similarity = 1.0 / (1.0 + distance) if distance != float('inf') else 0.0
        else:
            similarity = cosine_similarity(vector1, vector2)
        
        # Cache the result
        if use_cache:
            self._add_to_cache(cache_key, similarity)
            self.cache_misses += 1
        
        return similarity
    
    def _create_cache_key(self, vector1: List[float], vector2: List[float], method: str) -> str:
        """Create cache key from vectors"""
        # Simple hash-based key (not perfect but efficient)
        v1_hash = hash(tuple(vector1[:10]))  # Use first 10 elements for efficiency
        v2_hash = hash(tuple(vector2[:10]))
        return f"{method}_{min(v1_hash, v2_hash)}_{max(v1_hash, v2_hash)}"
    
    def _add_to_cache(self, key: str, value: float):
        """Add value to cache with size management"""
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = value
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache performance statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache),
            'max_cache_size': self.cache_size
        }
    
    def clear_cache(self):
        """Clear the cache"""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0