"""
Circuit Breaker Implementation for DNA Origami AutoEncoder

Provides circuit breaker patterns to prevent cascading failures and
protect system components from overload conditions.
"""

import asyncio
import time
from typing import Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import deque
import statistics

from ..utils.logger import get_logger

logger = get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Blocking requests
    HALF_OPEN = "half_open" # Testing recovery


@dataclass
class CircuitMetrics:
    """Metrics tracked by the circuit breaker."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeout_requests: int = 0
    
    # Recent history for sliding window
    recent_results: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Timestamps
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    circuit_opened_time: Optional[float] = None
    
    def success_rate(self, window_size: Optional[int] = None) -> float:
        """Calculate success rate over recent requests."""
        if window_size is None:
            recent_requests = list(self.recent_results)
        else:
            recent_requests = list(self.recent_results)[-window_size:]
            
        if not recent_requests:
            return 1.0
            
        successes = sum(1 for result in recent_requests if result)
        return successes / len(recent_requests)
        
    def average_response_time(self, window_size: Optional[int] = None) -> float:
        """Calculate average response time over recent requests."""
        if window_size is None:
            recent_times = list(self.recent_response_times)
        else:
            recent_times = list(self.recent_response_times)[-window_size:]
            
        if not recent_times:
            return 0.0
            
        return statistics.mean(recent_times)
        

class CircuitBreaker:
    """
    Circuit breaker implementation for protecting DNA origami system components.
    
    Features:
    - Configurable failure thresholds
    - Automatic recovery testing
    - Response time monitoring
    - Sliding window metrics
    - Custom failure detection
    - Graceful degradation support
    """
    
    def __init__(self,
                 name: str,
                 failure_threshold: float = 0.5,       # 50% failure rate
                 recovery_timeout: float = 60.0,       # 1 minute
                 request_timeout: float = 30.0,        # 30 seconds
                 minimum_requests: int = 10,           # Minimum requests before evaluation
                 half_open_max_calls: int = 3,        # Max calls in half-open state
                 slow_call_threshold: float = 5.0,    # 5 seconds
                 slow_call_rate_threshold: float = 0.8): # 80% slow calls
        
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.request_timeout = request_timeout
        self.minimum_requests = minimum_requests
        self.half_open_max_calls = half_open_max_calls
        self.slow_call_threshold = slow_call_threshold
        self.slow_call_rate_threshold = slow_call_rate_threshold
        
        # State management
        self.state = CircuitState.CLOSED
        self.metrics = CircuitMetrics()
        self._lock = threading.Lock()
        
        # Half-open state tracking
        self._half_open_calls = 0
        self._half_open_successes = 0
        
        # Custom failure detection
        self._failure_predicates: list = []
        self._custom_exception_types: tuple = ()
        
    def add_failure_predicate(self, predicate: Callable[[Any], bool]):
        """Add a custom predicate to detect failures."""
        self._failure_predicates.append(predicate)
        
    def add_exception_types(self, *exception_types):
        """Add exception types that should trigger circuit breaker."""
        self._custom_exception_types += exception_types
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function call protected by the circuit breaker."""
        # Check if circuit is open
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is open"
                )
                
        # Check if in half-open state and exceeded max calls
        if self.state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.half_open_max_calls:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' half-open call limit exceeded"
                )
                
        # Execute the function call
        start_time = time.time()
        success = False
        result = None
        error = None
        
        try:
            # Use asyncio timeout if function is async
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.request_timeout
                )
            else:
                # Use ThreadPoolExecutor for sync functions with timeout
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, func, *args, **kwargs),
                    timeout=self.request_timeout
                )
                
            # Check for custom failure conditions
            if self._is_failure(result):
                success = False
                error = "Custom failure condition met"
            else:
                success = True
                
        except asyncio.TimeoutError:
            success = False
            error = "Request timeout"
            self._record_timeout()
            
        except Exception as e:
            success = False
            error = str(e)
            
            # Check if this exception type should trigger circuit breaker
            if not self._should_handle_exception(e):
                # Re-raise exceptions that shouldn't trigger circuit breaker
                raise
                
        finally:
            response_time = time.time() - start_time
            self._record_call(success, response_time, error)
            
        if not success:
            raise CircuitBreakerError(f"Circuit breaker call failed: {error}")
            
        return result
        
    def _is_failure(self, result: Any) -> bool:
        """Check if result represents a failure using custom predicates."""
        return any(predicate(result) for predicate in self._failure_predicates)
        
    def _should_handle_exception(self, exception: Exception) -> bool:
        """Check if exception should trigger circuit breaker logic."""
        if not self._custom_exception_types:
            return True  # Handle all exceptions by default
            
        return isinstance(exception, self._custom_exception_types)
        
    def _record_call(self, success: bool, response_time: float, error: Optional[str] = None):
        """Record the result of a function call."""
        with self._lock:
            # Update metrics
            self.metrics.total_requests += 1
            
            if success:
                self.metrics.successful_requests += 1
                self.metrics.last_success_time = time.time()
                
                # Handle half-open state
                if self.state == CircuitState.HALF_OPEN:
                    self._half_open_successes += 1
                    
            else:
                self.metrics.failed_requests += 1
                self.metrics.last_failure_time = time.time()
                
            # Update recent history
            self.metrics.recent_results.append(success)
            self.metrics.recent_response_times.append(response_time)
            
            # Check for slow calls
            if response_time > self.slow_call_threshold:
                logger.warning(f"Slow call detected in {self.name}: {response_time:.2f}s")
                
            # Update half-open tracking
            if self.state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1
                
            # Evaluate circuit state
            self._evaluate_circuit_state()
            
    def _record_timeout(self):
        """Record a timeout event."""
        with self._lock:
            self.metrics.timeout_requests += 1
            
    def _evaluate_circuit_state(self):
        """Evaluate and potentially change circuit state based on metrics."""
        if self.state == CircuitState.CLOSED:
            self._evaluate_closed_state()
        elif self.state == CircuitState.HALF_OPEN:
            self._evaluate_half_open_state()
            
    def _evaluate_closed_state(self):
        """Evaluate whether to open the circuit from closed state."""
        # Need minimum requests before evaluation
        if self.metrics.total_requests < self.minimum_requests:
            return
            
        # Check failure rate
        failure_rate = 1.0 - self.metrics.success_rate()
        if failure_rate >= self.failure_threshold:
            self._open_circuit("High failure rate")
            return
            
        # Check slow call rate
        if len(self.metrics.recent_response_times) >= self.minimum_requests:
            slow_calls = sum(
                1 for time in self.metrics.recent_response_times
                if time > self.slow_call_threshold
            )
            slow_call_rate = slow_calls / len(self.metrics.recent_response_times)
            
            if slow_call_rate >= self.slow_call_rate_threshold:
                self._open_circuit("High slow call rate")
                
    def _evaluate_half_open_state(self):
        """Evaluate whether to close or open the circuit from half-open state."""
        if self._half_open_calls >= self.half_open_max_calls:
            success_rate = self._half_open_successes / self._half_open_calls
            
            if success_rate >= (1.0 - self.failure_threshold):
                self._close_circuit("Recovery successful")
            else:
                self._open_circuit("Recovery failed")
                
    def _open_circuit(self, reason: str):
        """Transition to open state."""
        logger.warning(f"Opening circuit breaker '{self.name}': {reason}")
        self.state = CircuitState.OPEN
        self.metrics.circuit_opened_time = time.time()
        
    def _close_circuit(self, reason: str):
        """Transition to closed state."""
        logger.info(f"Closing circuit breaker '{self.name}': {reason}")
        self.state = CircuitState.CLOSED
        self._reset_half_open_counters()
        
    def _transition_to_half_open(self):
        """Transition to half-open state."""
        logger.info(f"Transitioning circuit breaker '{self.name}' to half-open")
        self.state = CircuitState.HALF_OPEN
        self._reset_half_open_counters()
        
    def _reset_half_open_counters(self):
        """Reset half-open state counters."""
        self._half_open_calls = 0
        self._half_open_successes = 0
        
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.metrics.circuit_opened_time is None:
            return True
            
        time_since_opened = time.time() - self.metrics.circuit_opened_time
        return time_since_opened >= self.recovery_timeout
        
    def force_open(self, reason: str = "Manually opened"):
        """Manually force the circuit to open state."""
        with self._lock:
            self._open_circuit(reason)
            
    def force_close(self, reason: str = "Manually closed"):
        """Manually force the circuit to closed state."""
        with self._lock:
            self._close_circuit(reason)
            
    def reset_metrics(self):
        """Reset all metrics."""
        with self._lock:
            self.metrics = CircuitMetrics()
            self._reset_half_open_counters()
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get current circuit breaker metrics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "timeout_requests": self.metrics.timeout_requests,
                "success_rate": self.metrics.success_rate(),
                "failure_rate": 1.0 - self.metrics.success_rate(),
                "average_response_time": self.metrics.average_response_time(),
                "last_failure_time": self.metrics.last_failure_time,
                "last_success_time": self.metrics.last_success_time,
                "circuit_opened_time": self.metrics.circuit_opened_time,
                "half_open_calls": self._half_open_calls,
                "half_open_successes": self._half_open_successes,
                "configuration": {
                    "failure_threshold": self.failure_threshold,
                    "recovery_timeout": self.recovery_timeout,
                    "request_timeout": self.request_timeout,
                    "minimum_requests": self.minimum_requests,
                    "half_open_max_calls": self.half_open_max_calls,
                    "slow_call_threshold": self.slow_call_threshold,
                    "slow_call_rate_threshold": self.slow_call_rate_threshold
                }
            }
            

class CircuitBreakerError(Exception):
    """Base exception for circuit breaker errors."""
    pass


class CircuitBreakerOpenError(CircuitBreakerError):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreakerManager:
    """
    Manages multiple circuit breakers for different system components.
    """
    
    def __init__(self):
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        
    def create_circuit_breaker(self, name: str, **kwargs) -> CircuitBreaker:
        """Create a new circuit breaker."""
        if name in self._circuit_breakers:
            raise ValueError(f"Circuit breaker '{name}' already exists")
            
        circuit_breaker = CircuitBreaker(name, **kwargs)
        self._circuit_breakers[name] = circuit_breaker
        
        logger.info(f"Created circuit breaker: {name}")
        return circuit_breaker
        
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name."""
        return self._circuit_breakers.get(name)
        
    def remove_circuit_breaker(self, name: str) -> bool:
        """Remove a circuit breaker."""
        if name in self._circuit_breakers:
            del self._circuit_breakers[name]
            logger.info(f"Removed circuit breaker: {name}")
            return True
        return False
        
    async def call_with_circuit_breaker(self, 
                                       name: str,
                                       func: Callable,
                                       *args,
                                       **kwargs) -> Any:
        """Call a function using the specified circuit breaker."""
        circuit_breaker = self.get_circuit_breaker(name)
        if not circuit_breaker:
            raise ValueError(f"Circuit breaker '{name}' not found")
            
        return await circuit_breaker.call(func, *args, **kwargs)
        
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all circuit breakers."""
        return {
            name: breaker.get_metrics()
            for name, breaker in self._circuit_breakers.items()
        }
        
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status of all circuit breakers."""
        total_breakers = len(self._circuit_breakers)
        open_breakers = sum(
            1 for breaker in self._circuit_breakers.values()
            if breaker.state == CircuitState.OPEN
        )
        half_open_breakers = sum(
            1 for breaker in self._circuit_breakers.values()
            if breaker.state == CircuitState.HALF_OPEN
        )
        closed_breakers = total_breakers - open_breakers - half_open_breakers
        
        return {
            "total_circuit_breakers": total_breakers,
            "open": open_breakers,
            "half_open": half_open_breakers,
            "closed": closed_breakers,
            "overall_health": "healthy" if open_breakers == 0 else "degraded" if open_breakers < total_breakers else "critical"
        }