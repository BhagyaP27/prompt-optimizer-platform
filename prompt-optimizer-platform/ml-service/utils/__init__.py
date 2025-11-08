"""Utility functions for ML service"""

from .preprocessing import clean_text, tokenize_text
from .cache import get_cached_prediction, cache_prediction
from .metrics import track_prediction, get_metrics_summary

__all__ = [
    'clean_text',
    'tokenize_text',
    'get_cached_prediction',
    'cache_prediction',
    'track_prediction',
    'get_metrics_summary'
]


# ============================================
# ml-service/utils/preprocessing.py
# ============================================
"""
Text preprocessing utilities
Used before sending text to model
"""

import re
from typing import Optional


def clean_text(text: str) -> str:
    """
    Clean and normalize input text
    
    Steps:
    1. Remove excessive whitespace
    2. Normalize unicode characters
    3. Remove control characters
    4. Trim to reasonable length
    
    Args:
        text: Raw input text
    
    Returns:
        Cleaned text
    """
    # Remove control characters
    text = ''.join(char for char in text if char.isprintable() or char.isspace())
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Remove excessive punctuation (more than 3 in a row)
    text = re.sub(r'([!?.]){4,}', r'\1\1\1', text)
    
    # Trim
    text = text.strip()
    
    return text


def tokenize_text(text: str, tokenizer, max_length: int = 256) -> dict:
    """
    Tokenize text for model input
    
    Args:
        text: Input text to tokenize
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
    
    Returns:
        Dictionary with input_ids and attention_mask
    """
    # Add task prefix for T5
    text = f"optimize prompt: {text}"
    
    # Tokenize
    encoded = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    
    return encoded


def sanitize_prompt(prompt: str) -> str:
    """
    Sanitize prompt for security
    Remove potentially dangerous patterns
    
    Args:
        prompt: User input prompt
    
    Returns:
        Sanitized prompt
    
    Raises:
        ValueError: If dangerous pattern detected
    """
    # List of dangerous patterns
    dangerous_patterns = [
        r'<script[^>]*>.*?</script>',  # Script tags
        r'javascript:',  # JavaScript protocol
        r'on\w+\s*=',  # Event handlers (onclick, onerror, etc.)
        r'<iframe',  # Iframe injection
        r'data:text/html',  # Data URI
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, prompt, re.IGNORECASE):
            raise ValueError(f"Potentially dangerous pattern detected")
    
    return prompt


def truncate_text(text: str, max_length: int = 1000) -> str:
    """
    Truncate text to maximum length
    Tries to break at sentence boundary
    
    Args:
        text: Text to truncate
        max_length: Maximum length in characters
    
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    # Try to truncate at sentence boundary
    truncated = text[:max_length]
    
    # Find last sentence end
    last_period = truncated.rfind('.')
    last_question = truncated.rfind('?')
    last_exclamation = truncated.rfind('!')
    
    last_sentence = max(last_period, last_question, last_exclamation)
    
    if last_sentence > max_length * 0.8:  # If we found a sentence close to end
        return truncated[:last_sentence + 1]
    
    # Otherwise, just truncate and add ellipsis
    return truncated.rstrip() + '...'


# ============================================
# ml-service/utils/cache.py
# ============================================
"""
Caching utilities using Redis
Reduces load on ML model for repeated requests
"""

import hashlib
import json
from typing import Optional
import redis
from config import settings


# Initialize Redis client (None if caching disabled)
redis_client = None
if settings.enable_cache and settings.redis_url:
    try:
        redis_client = redis.from_url(settings.redis_url, decode_responses=True)
        redis_client.ping()  # Test connection
    except Exception as e:
        print(f"Redis connection failed: {e}")
        redis_client = None


def get_cache_key(prompt: str, params: dict) -> str:
    """
    Generate cache key from prompt and parameters
    
    Same prompt with same parameters should produce same key
    Different parameters (temperature, etc.) produce different keys
    
    Args:
        prompt: User prompt
        params: Generation parameters
    
    Returns:
        Cache key (hash)
    """
    # Create string representation of prompt + params
    cache_string = f"{prompt}:{json.dumps(params, sort_keys=True)}"
    
    # Generate hash
    cache_hash = hashlib.md5(cache_string.encode()).hexdigest()
    
    return f"prompt:{cache_hash}"


def get_cached_prediction(prompt: str, params: dict) -> Optional[str]:
    """
    Get prediction from cache if available
    
    Args:
        prompt: User prompt
        params: Generation parameters
    
    Returns:
        Cached prediction or None if not found
    """
    if not redis_client:
        return None
    
    try:
        cache_key = get_cache_key(prompt, params)
        cached = redis_client.get(cache_key)
        
        if cached:
            return json.loads(cached)
        
        return None
    
    except Exception as e:
        # Don't fail request if cache fails
        print(f"Cache read error: {e}")
        return None


def cache_prediction(prompt: str, params: dict, prediction: str) -> bool:
    """
    Cache a prediction
    
    Args:
        prompt: User prompt
        params: Generation parameters
        prediction: Model prediction to cache
    
    Returns:
        True if cached successfully, False otherwise
    """
    if not redis_client:
        return False
    
    try:
        cache_key = get_cache_key(prompt, params)
        
        # Cache with TTL (time-to-live)
        redis_client.setex(
            cache_key,
            settings.cache_ttl,
            json.dumps(prediction)
        )
        
        return True
    
    except Exception as e:
        # Don't fail request if cache fails
        print(f"Cache write error: {e}")
        return False


def clear_cache() -> bool:
    """
    Clear all cached predictions
    Useful after model update
    
    Returns:
        True if successful
    """
    if not redis_client:
        return False
    
    try:
        # Find all keys with our prefix
        keys = redis_client.keys("prompt:*")
        
        if keys:
            redis_client.delete(*keys)
        
        return True
    
    except Exception as e:
        print(f"Cache clear error: {e}")
        return False


# ============================================
# ml-service/utils/metrics.py
# ============================================
"""
Metrics tracking and aggregation
Works with Prometheus for monitoring
"""

from collections import defaultdict
from typing import Dict
import time


# In-memory metrics storage (Prometheus handles persistence)
_metrics = {
    'predictions': defaultdict(int),
    'latencies': [],
    'errors': defaultdict(int),
    'start_time': time.time()
}


def track_prediction(
    status: str,
    latency_ms: float,
    model_version: str
):
    """
    Track a prediction event
    
    Args:
        status: 'success' or 'error'
        latency_ms: Processing time in milliseconds
        model_version: Version of model used
    """
    # Increment counters
    _metrics['predictions'][status] += 1
    
    # Track latency
    _metrics['latencies'].append(latency_ms)
    
    # Keep only last 1000 latencies
    if len(_metrics['latencies']) > 1000:
        _metrics['latencies'] = _metrics['latencies'][-1000:]


def get_metrics_summary() -> Dict:
    """
    Get summary of metrics
    
    Returns:
        Dictionary with aggregated metrics
    """
    total_predictions = sum(_metrics['predictions'].values())
    success_count = _metrics['predictions'].get('success', 0)
    error_count = _metrics['predictions'].get('error', 0)
    
    latencies = _metrics['latencies']
    
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        sorted_latencies = sorted(latencies)
        p95_latency = sorted_latencies[int(len(sorted_latencies) * 0.95)]
        p99_latency = sorted_latencies[int(len(sorted_latencies) * 0.99)]
    else:
        avg_latency = 0
        p95_latency = 0
        p99_latency = 0
    
    uptime = time.time() - _metrics['start_time']
    
    return {
        'total_predictions': total_predictions,
        'success_count': success_count,
        'error_count': error_count,
        'success_rate': success_count / total_predictions if total_predictions > 0 else 0,
        'average_latency_ms': avg_latency,
        'p95_latency_ms': p95_latency,
        'p99_latency_ms': p99_latency,
        'uptime_seconds': uptime
    }


def reset_metrics():
    """Reset all metrics (useful for testing)"""
    _metrics['predictions'].clear()
    _metrics['latencies'].clear()
    _metrics['errors'].clear()
    _metrics['start_time'] = time.time()