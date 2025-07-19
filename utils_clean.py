# utils.py - Performance utilities
import functools
import time
import logging
from typing import Callable, Any
import gc

def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('extraction.log')
        ]
    )
    return logging.getLogger(__name__)

def measure_time(func: Callable) -> Callable:
    """Decorator to measure function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger = logging.getLogger(func.__module__)
        logger.debug(f"{func.__name__} took {end_time - start_time:.3f}s")
        
        return result
    return wrapper

class PerformanceOptimizer:
    """Performance optimization utilities"""
    
    @staticmethod
    def optimize_memory():
        """Force garbage collection to free memory"""
        gc.collect()
    
    @staticmethod
    def cache_results(maxsize=128):
        """LRU cache decorator for expensive operations"""
        def decorator(func):
            cache = {}
            cache_order = []
            
            @functools.wraps(func)
            def wrapper(*args):
                key = str(args)
                
                if key in cache:
                    # Move to end (most recently used)
                    cache_order.remove(key)
                    cache_order.append(key)
                    return cache[key]
                
                # Compute result
                result = func(*args)
                
                # Add to cache
                cache[key] = result
                cache_order.append(key)
                
                # Remove oldest if cache full
                if len(cache) > maxsize:
                    oldest = cache_order.pop(0)
                    del cache[oldest]
                
                return result
            
            return wrapper
        return decorator
</content>
</invoke>
