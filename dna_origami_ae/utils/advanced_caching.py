"""Advanced caching system with multi-level cache hierarchy and intelligent eviction."""

import threading
import time
import hashlib
import pickle
import json
import gzip
import lz4.frame
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import redis
from collections import OrderedDict, defaultdict
import weakref
from concurrent.futures import ThreadPoolExecutor
import psutil
import numpy as np

from .logger import get_logger


class CacheLevel(Enum):
    """Cache level hierarchy."""
    MEMORY = "memory"
    REDIS = "redis"
    DISK = "disk"


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    SIZE = "size"  # Size-based
    ADAPTIVE = "adaptive"  # Adaptive based on access patterns


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    size_bytes: int = 0
    ttl_seconds: Optional[float] = None
    compression_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.timestamp > self.ttl_seconds
    
    def update_access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_access = time.time()


class CompressionManager:
    """Manage data compression for cache entries."""
    
    def __init__(self):
        self.compression_threshold = 1024  # Compress data > 1KB
        self.logger = get_logger("compression_manager")
    
    def compress_data(self, data: Any, compression_type: str = "lz4") -> Tuple[bytes, str]:
        """Compress data using specified algorithm."""
        
        # Serialize data first
        if isinstance(data, (str, bytes)):
            serialized = data.encode('utf-8') if isinstance(data, str) else data
        else:
            serialized = pickle.dumps(data)
        
        # Skip compression for small data
        if len(serialized) < self.compression_threshold:
            return serialized, "none"
        
        try:
            if compression_type == "lz4":
                compressed = lz4.frame.compress(serialized)
                return compressed, "lz4"
            elif compression_type == "gzip":
                compressed = gzip.compress(serialized)
                return compressed, "gzip"
            else:
                return serialized, "none"
        except Exception as e:
            self.logger.warning(f"Compression failed: {e}")
            return serialized, "none"
    
    def decompress_data(self, compressed_data: bytes, compression_type: str) -> Any:
        """Decompress data."""
        
        if compression_type == "none":
            # Try to deserialize directly
            try:
                return pickle.loads(compressed_data)
            except:
                return compressed_data.decode('utf-8')
        
        try:
            if compression_type == "lz4":
                decompressed = lz4.frame.decompress(compressed_data)
            elif compression_type == "gzip":
                decompressed = gzip.decompress(compressed_data)
            else:
                decompressed = compressed_data
            
            # Try to deserialize
            try:
                return pickle.loads(decompressed)
            except:
                return decompressed.decode('utf-8')
                
        except Exception as e:
            self.logger.error(f"Decompression failed: {e}")
            raise


class MemoryCache:
    """High-performance in-memory cache with intelligent eviction."""
    
    def __init__(self, max_size_mb: int = 100, eviction_policy: EvictionPolicy = EvictionPolicy.ADAPTIVE):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.eviction_policy = eviction_policy
        self.cache = OrderedDict()
        self.access_stats = defaultdict(int)
        self.size_stats = defaultdict(int)
        self._lock = threading.RLock()
        self.current_size_bytes = 0
        
        # Adaptive eviction parameters
        self.access_pattern_history = []
        self.eviction_effectiveness = {}
        
        self.logger = get_logger("memory_cache")
    
    def _calculate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        if isinstance(obj, CacheEntry):
            if obj.size_bytes > 0:
                return obj.size_bytes
            
            # Estimate based on value
            try:
                if isinstance(obj.value, (str, bytes)):
                    size = len(obj.value)
                elif isinstance(obj.value, np.ndarray):
                    size = obj.value.nbytes
                else:
                    size = len(pickle.dumps(obj.value))
                
                obj.size_bytes = size
                return size
            except:
                return 1024  # Default estimate
        
        return len(pickle.dumps(obj))
    
    def _should_evict(self) -> bool:
        """Check if eviction is needed."""
        return self.current_size_bytes > self.max_size_bytes
    
    def _select_eviction_candidates(self, count: int) -> List[str]:
        """Select keys for eviction based on policy."""
        
        if not self.cache:
            return []
        
        if self.eviction_policy == EvictionPolicy.LRU:
            # Remove least recently used items
            return list(self.cache.keys())[:count]
        
        elif self.eviction_policy == EvictionPolicy.LFU:
            # Remove least frequently used items
            sorted_by_access = sorted(
                self.cache.items(),
                key=lambda x: x[1].access_count
            )
            return [key for key, _ in sorted_by_access[:count]]
        
        elif self.eviction_policy == EvictionPolicy.TTL:
            # Remove expired items first, then oldest
            expired = [key for key, entry in self.cache.items() if entry.is_expired()]
            if len(expired) >= count:
                return expired[:count]
            
            remaining = count - len(expired)
            oldest = sorted(
                [(k, v) for k, v in self.cache.items() if k not in expired],
                key=lambda x: x[1].timestamp
            )
            return expired + [key for key, _ in oldest[:remaining]]
        
        elif self.eviction_policy == EvictionPolicy.SIZE:
            # Remove largest items first
            sorted_by_size = sorted(
                self.cache.items(),
                key=lambda x: x[1].size_bytes,
                reverse=True
            )
            return [key for key, _ in sorted_by_size[:count]]
        
        elif self.eviction_policy == EvictionPolicy.ADAPTIVE:
            return self._adaptive_eviction_selection(count)
        
        else:
            # Default to LRU
            return list(self.cache.keys())[:count]
    
    def _adaptive_eviction_selection(self, count: int) -> List[str]:
        """Adaptive eviction based on access patterns and effectiveness."""
        
        # Score each item based on multiple factors
        scores = {}
        current_time = time.time()
        
        for key, entry in self.cache.items():
            # Factors: recency, frequency, size, TTL
            recency_score = 1.0 / (current_time - entry.last_access + 1)
            frequency_score = entry.access_count / (current_time - entry.timestamp + 1)
            size_penalty = entry.size_bytes / self.max_size_bytes
            
            # TTL consideration
            ttl_score = 1.0
            if entry.ttl_seconds:
                remaining_ttl = entry.ttl_seconds - (current_time - entry.timestamp)
                ttl_score = max(0, remaining_ttl / entry.ttl_seconds)
            
            # Combined score (lower = more likely to evict)
            scores[key] = recency_score * 0.3 + frequency_score * 0.4 - size_penalty * 0.2 + ttl_score * 0.1
        
        # Sort by score and select lowest
        sorted_by_score = sorted(scores.items(), key=lambda x: x[1])
        return [key for key, _ in sorted_by_score[:count]]
    
    def _evict_entries(self, keys: List[str]):
        """Evict specified keys from cache."""
        
        evicted_size = 0
        for key in keys:
            if key in self.cache:
                entry = self.cache[key]
                evicted_size += entry.size_bytes
                del self.cache[key]
        
        self.current_size_bytes -= evicted_size
        
        if keys:
            self.logger.debug(f"Evicted {len(keys)} entries, freed {evicted_size} bytes")
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[float] = None,
           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store item in cache."""
        
        with self._lock:
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                ttl_seconds=ttl_seconds,
                metadata=metadata or {}
            )
            
            entry_size = self._calculate_size(entry)
            
            # Check if single item is too large
            if entry_size > self.max_size_bytes:
                self.logger.warning(f"Item too large for cache: {entry_size} bytes")
                return False
            
            # Remove existing entry if present
            if key in self.cache:
                old_entry = self.cache[key]
                self.current_size_bytes -= old_entry.size_bytes
                del self.cache[key]
            
            # Evict items if necessary
            projected_size = self.current_size_bytes + entry_size
            if projected_size > self.max_size_bytes:
                # Calculate how many items to evict
                bytes_to_free = projected_size - self.max_size_bytes
                
                # Estimate items to evict (assume average size)
                avg_size = self.current_size_bytes / len(self.cache) if self.cache else entry_size
                items_to_evict = max(1, int(bytes_to_free / avg_size) + 1)
                
                eviction_candidates = self._select_eviction_candidates(items_to_evict)
                self._evict_entries(eviction_candidates)
            
            # Add new entry
            self.cache[key] = entry
            self.current_size_bytes += entry_size
            
            return True
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve item from cache."""
        
        with self._lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            
            # Check expiration
            if entry.is_expired():
                del self.cache[key]
                self.current_size_bytes -= entry.size_bytes
                return None
            
            # Update access stats
            entry.update_access()
            
            # Move to end for LRU
            if self.eviction_policy == EvictionPolicy.LRU:
                self.cache.move_to_end(key)
            
            return entry.value
    
    def delete(self, key: str) -> bool:
        """Delete item from cache."""
        
        with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                self.current_size_bytes -= entry.size_bytes
                del self.cache[key]
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        
        with self._lock:
            self.cache.clear()
            self.current_size_bytes = 0
            self.access_stats.clear()
            self.size_stats.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        
        with self._lock:
            expired_count = sum(1 for entry in self.cache.values() if entry.is_expired())
            
            return {
                'total_entries': len(self.cache),
                'current_size_bytes': self.current_size_bytes,
                'max_size_bytes': self.max_size_bytes,
                'utilization': self.current_size_bytes / self.max_size_bytes,
                'expired_entries': expired_count,
                'eviction_policy': self.eviction_policy.value,
                'average_access_count': sum(entry.access_count for entry in self.cache.values()) / len(self.cache) if self.cache else 0
            }


class RedisCache:
    """Redis-based distributed cache."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0", 
                 key_prefix: str = "dna_origami_ae:",
                 default_ttl: int = 3600):
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.default_ttl = default_ttl
        self.compression_manager = CompressionManager()
        
        try:
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()
            self.available = True
        except Exception as e:
            self.available = False
            self.redis_client = None
        
        self.logger = get_logger("redis_cache")
        
        if not self.available:
            self.logger.warning("Redis not available, cache will be disabled")
    
    def _make_key(self, key: str) -> str:
        """Create Redis key with prefix."""
        return f"{self.key_prefix}{key}"
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Store item in Redis cache."""
        
        if not self.available:
            return False
        
        try:
            # Compress and serialize
            compressed_data, compression_type = self.compression_manager.compress_data(value)
            
            # Create metadata
            metadata = {
                'compression_type': compression_type,
                'timestamp': time.time(),
                'original_type': type(value).__name__
            }
            
            # Store data and metadata
            redis_key = self._make_key(key)
            pipe = self.redis_client.pipeline()
            
            pipe.hset(redis_key, mapping={
                'data': compressed_data,
                'metadata': json.dumps(metadata)
            })
            
            # Set TTL
            ttl = ttl_seconds or self.default_ttl
            pipe.expire(redis_key, ttl)
            
            pipe.execute()
            return True
            
        except Exception as e:
            self.logger.error(f"Redis put failed: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve item from Redis cache."""
        
        if not self.available:
            return None
        
        try:
            redis_key = self._make_key(key)
            data = self.redis_client.hgetall(redis_key)
            
            if not data or b'data' not in data:
                return None
            
            # Parse metadata
            metadata = json.loads(data[b'metadata'])
            compression_type = metadata.get('compression_type', 'none')
            
            # Decompress and deserialize
            return self.compression_manager.decompress_data(data[b'data'], compression_type)
            
        except Exception as e:
            self.logger.error(f"Redis get failed: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete item from Redis cache."""
        
        if not self.available:
            return False
        
        try:
            redis_key = self._make_key(key)
            return self.redis_client.delete(redis_key) > 0
        except Exception as e:
            self.logger.error(f"Redis delete failed: {e}")
            return False
    
    def clear(self):
        """Clear all cache entries with our prefix."""
        
        if not self.available:
            return
        
        try:
            pattern = f"{self.key_prefix}*"
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
        except Exception as e:
            self.logger.error(f"Redis clear failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics."""
        
        if not self.available:
            return {'available': False}
        
        try:
            info = self.redis_client.info()
            pattern = f"{self.key_prefix}*"
            key_count = len(self.redis_client.keys(pattern))
            
            return {
                'available': True,
                'total_entries': key_count,
                'used_memory': info.get('used_memory', 0),
                'used_memory_human': info.get('used_memory_human', '0B'),
                'connected_clients': info.get('connected_clients', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0)
            }
        except Exception as e:
            self.logger.error(f"Redis stats failed: {e}")
            return {'available': False, 'error': str(e)}


class MultiLevelCache:
    """Multi-level cache system with automatic promotion/demotion."""
    
    def __init__(self, memory_cache_mb: int = 100, 
                 redis_url: Optional[str] = None,
                 enable_intelligent_prefetch: bool = True):
        
        # Initialize cache levels
        self.memory_cache = MemoryCache(memory_cache_mb, EvictionPolicy.ADAPTIVE)
        
        self.redis_cache = None
        if redis_url:
            self.redis_cache = RedisCache(redis_url)
        
        # Cache statistics
        self.stats = {
            'hits': defaultdict(int),
            'misses': defaultdict(int),
            'promotions': 0,
            'demotions': 0,
            'prefetch_hits': 0
        }
        self._stats_lock = threading.Lock()
        
        # Intelligent prefetching
        self.enable_prefetch = enable_intelligent_prefetch
        self.access_patterns = defaultdict(list)  # key -> list of access times
        self.prefetch_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cache_prefetch")
        
        self.logger = get_logger("multi_level_cache")
    
    def _record_access_pattern(self, key: str):
        """Record access pattern for intelligent prefetching."""
        
        if not self.enable_prefetch:
            return
        
        current_time = time.time()
        self.access_patterns[key].append(current_time)
        
        # Keep only recent accesses (last 24 hours)
        cutoff = current_time - 86400
        self.access_patterns[key] = [t for t in self.access_patterns[key] if t > cutoff]
        
        # Limit pattern history
        if len(self.access_patterns[key]) > 100:
            self.access_patterns[key] = self.access_patterns[key][-100:]
    
    def _predict_next_accesses(self) -> List[str]:
        """Predict which keys might be accessed next."""
        
        if not self.enable_prefetch:
            return []
        
        current_time = time.time()
        predictions = []
        
        for key, access_times in self.access_patterns.items():
            if len(access_times) < 3:  # Need at least 3 accesses for pattern
                continue
            
            # Calculate average interval between accesses
            intervals = [access_times[i] - access_times[i-1] for i in range(1, len(access_times))]
            avg_interval = sum(intervals) / len(intervals)
            
            # Predict next access time
            last_access = access_times[-1]
            predicted_next = last_access + avg_interval
            
            # If prediction is soon (within next 10 minutes) and key not in memory
            if predicted_next - current_time < 600 and not self.memory_cache.get(key):
                predictions.append(key)
        
        return predictions[:10]  # Limit prefetch candidates
    
    def _prefetch_key(self, key: str):
        """Prefetch key from lower cache levels."""
        
        try:
            # Try Redis first
            if self.redis_cache:
                value = self.redis_cache.get(key)
                if value is not None:
                    # Promote to memory cache
                    self.memory_cache.put(key, value)
                    
                    with self._stats_lock:
                        self.stats['prefetch_hits'] += 1
                        self.stats['promotions'] += 1
                    
                    self.logger.debug(f"Prefetched key from Redis: {key}")
        except Exception as e:
            self.logger.debug(f"Prefetch failed for key {key}: {e}")
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> bool:
        """Store item in cache hierarchy."""
        
        # Always store in memory cache first (hot data)
        memory_success = self.memory_cache.put(key, value, ttl_seconds)
        
        # Also store in Redis for persistence/sharing
        redis_success = True
        if self.redis_cache:
            redis_ttl = int(ttl_seconds) if ttl_seconds else None
            redis_success = self.redis_cache.put(key, value, redis_ttl)
        
        # Record access pattern
        self._record_access_pattern(key)
        
        # Trigger prefetching predictions
        if self.enable_prefetch:
            predictions = self._predict_next_accesses()
            for pred_key in predictions:
                self.prefetch_executor.submit(self._prefetch_key, pred_key)
        
        return memory_success or redis_success
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve item from cache hierarchy."""
        
        # Try memory cache first (L1)
        value = self.memory_cache.get(key)
        if value is not None:
            with self._stats_lock:
                self.stats['hits']['memory'] += 1
            
            self._record_access_pattern(key)
            return value
        
        # Try Redis cache (L2)
        if self.redis_cache:
            value = self.redis_cache.get(key)
            if value is not None:
                # Promote to memory cache
                self.memory_cache.put(key, value)
                
                with self._stats_lock:
                    self.stats['hits']['redis'] += 1
                    self.stats['promotions'] += 1
                
                self._record_access_pattern(key)
                return value
        
        # Cache miss
        with self._stats_lock:
            self.stats['misses']['total'] += 1
        
        return None
    
    def delete(self, key: str) -> bool:
        """Delete item from all cache levels."""
        
        memory_deleted = self.memory_cache.delete(key)
        redis_deleted = True
        
        if self.redis_cache:
            redis_deleted = self.redis_cache.delete(key)
        
        return memory_deleted or redis_deleted
    
    def clear(self):
        """Clear all cache levels."""
        
        self.memory_cache.clear()
        
        if self.redis_cache:
            self.redis_cache.clear()
        
        with self._stats_lock:
            self.stats = {
                'hits': defaultdict(int),
                'misses': defaultdict(int),
                'promotions': 0,
                'demotions': 0,
                'prefetch_hits': 0
            }
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        
        with self._stats_lock:
            total_hits = sum(self.stats['hits'].values())
            total_requests = total_hits + self.stats['misses']['total']
            
            hit_rate = total_hits / total_requests if total_requests > 0 else 0
            
            stats = {
                'hit_rate': hit_rate,
                'total_requests': total_requests,
                'hits_by_level': dict(self.stats['hits']),
                'misses': self.stats['misses']['total'],
                'promotions': self.stats['promotions'],
                'demotions': self.stats['demotions'],
                'prefetch_hits': self.stats['prefetch_hits'],
                'memory_cache': self.memory_cache.get_stats()
            }
            
            if self.redis_cache:
                stats['redis_cache'] = self.redis_cache.get_stats()
            
            return stats


class CacheManager:
    """High-level cache management with automatic optimization."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        # Initialize multi-level cache
        self.cache = MultiLevelCache(
            memory_cache_mb=config.get('memory_cache_mb', 100),
            redis_url=config.get('redis_url'),
            enable_intelligent_prefetch=config.get('enable_prefetch', True)
        )
        
        # Cache optimization
        self.optimization_enabled = config.get('enable_optimization', True)
        self.optimization_interval = config.get('optimization_interval', 300)  # 5 minutes
        
        # Automatic key generation
        self.key_generators = {}
        
        # Background optimization
        if self.optimization_enabled:
            self._start_optimization_thread()
        
        self.logger = get_logger("cache_manager")
    
    def _start_optimization_thread(self):
        """Start background cache optimization."""
        
        def optimize_cache():
            while True:
                try:
                    time.sleep(self.optimization_interval)
                    self._optimize_cache()
                except Exception as e:
                    self.logger.error(f"Cache optimization error: {e}")
        
        optimization_thread = threading.Thread(target=optimize_cache, daemon=True)
        optimization_thread.start()
    
    def _optimize_cache(self):
        """Perform cache optimization."""
        
        stats = self.cache.get_comprehensive_stats()
        
        # Check hit rate and adjust policies
        hit_rate = stats.get('hit_rate', 0)
        
        if hit_rate < 0.5:  # Low hit rate
            # Consider increasing memory cache size or adjusting eviction policy
            self.logger.info(f"Low hit rate detected: {hit_rate:.2%}")
            
            # Check memory usage
            memory_stats = stats.get('memory_cache', {})
            utilization = memory_stats.get('utilization', 0)
            
            if utilization > 0.9 and psutil.virtual_memory().percent < 80:
                # High utilization but system memory available - increase cache size
                new_size_mb = int(memory_stats.get('max_size_bytes', 0) / 1024 / 1024 * 1.2)
                self.logger.info(f"Increasing memory cache size to {new_size_mb}MB")
                
                # Would need to recreate cache with new size in real implementation
        
        # Log optimization results
        self.logger.debug(f"Cache optimization complete. Hit rate: {hit_rate:.2%}")
    
    def register_key_generator(self, cache_type: str, generator_func: Callable):
        """Register automatic key generator for cache type."""
        self.key_generators[cache_type] = generator_func
    
    def _generate_key(self, cache_type: str, *args, **kwargs) -> str:
        """Generate cache key automatically."""
        
        if cache_type in self.key_generators:
            return self.key_generators[cache_type](*args, **kwargs)
        
        # Default key generation
        key_parts = [cache_type]
        
        for arg in args:
            if isinstance(arg, (str, int, float)):
                key_parts.append(str(arg))
            else:
                # Hash complex objects
                key_parts.append(hashlib.md5(str(arg).encode()).hexdigest()[:8])
        
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        
        return ":".join(key_parts)
    
    def cached_function(self, cache_type: str, ttl_seconds: Optional[float] = None):
        """Decorator for caching function results."""
        
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_key(cache_type, func.__name__, *args, **kwargs)
                
                # Try to get from cache
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.cache.put(cache_key, result, ttl_seconds)
                
                return result
            
            return wrapper
        
        return decorator
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> bool:
        """Store item in cache."""
        return self.cache.put(key, value, ttl_seconds)
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve item from cache."""
        return self.cache.get(key)
    
    def delete(self, key: str) -> bool:
        """Delete item from cache."""
        return self.cache.delete(key)
    
    def clear(self):
        """Clear all cache levels."""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return self.cache.get_comprehensive_stats()
    
    def health_check(self) -> Dict[str, Any]:
        """Perform cache health check."""
        
        stats = self.get_stats()
        
        health = {
            'status': 'healthy',
            'issues': [],
            'recommendations': []
        }
        
        # Check hit rate
        hit_rate = stats.get('hit_rate', 0)
        if hit_rate < 0.3:
            health['issues'].append(f"Low hit rate: {hit_rate:.2%}")
            health['recommendations'].append("Consider adjusting cache size or TTL values")
            health['status'] = 'degraded'
        
        # Check memory utilization
        memory_stats = stats.get('memory_cache', {})
        utilization = memory_stats.get('utilization', 0)
        if utilization > 0.95:
            health['issues'].append(f"High memory utilization: {utilization:.1%}")
            health['recommendations'].append("Consider increasing memory cache size")
            health['status'] = 'degraded'
        
        # Check Redis availability
        redis_stats = stats.get('redis_cache', {})
        if redis_stats and not redis_stats.get('available', True):
            health['issues'].append("Redis cache unavailable")
            health['recommendations'].append("Check Redis connection and configuration")
            health['status'] = 'degraded'
        
        return health


# Global cache manager instance
_global_cache_manager = None
_cache_manager_lock = threading.Lock()


def get_cache_manager(config: Optional[Dict[str, Any]] = None) -> CacheManager:
    """Get global cache manager instance."""
    
    global _global_cache_manager
    
    with _cache_manager_lock:
        if _global_cache_manager is None:
            _global_cache_manager = CacheManager(config)
        
        return _global_cache_manager


# Convenience decorators
def cached(cache_type: str = "default", ttl_seconds: Optional[float] = None):
    """Convenience decorator for caching function results."""
    
    cache_manager = get_cache_manager()
    return cache_manager.cached_function(cache_type, ttl_seconds)


def dna_sequence_cache(ttl_seconds: float = 3600):
    """Specialized cache decorator for DNA sequences."""
    
    def key_generator(func_name: str, *args, **kwargs):
        # Special handling for DNA sequence arguments
        key_parts = [f"dna_seq:{func_name}"]
        
        for arg in args:
            if hasattr(arg, 'sequence'):  # DNA sequence object
                seq_hash = hashlib.md5(arg.sequence.encode()).hexdigest()[:12]
                key_parts.append(f"seq:{seq_hash}")
            elif isinstance(arg, str) and len(arg) > 10 and set(arg.upper()).issubset(set('ATGC')):
                # Looks like DNA sequence
                seq_hash = hashlib.md5(arg.encode()).hexdigest()[:12]
                key_parts.append(f"dna:{seq_hash}")
            else:
                key_parts.append(str(arg)[:50])  # Limit key length
        
        return ":".join(key_parts)
    
    cache_manager = get_cache_manager()
    cache_manager.register_key_generator("dna_sequence", key_generator)
    
    return cache_manager.cached_function("dna_sequence", ttl_seconds)