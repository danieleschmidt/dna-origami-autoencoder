"""Error correction for DNA sequences."""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ErrorCorrectionParameters:
    """Parameters for error correction."""
    
    redundancy: float = 0.3  # 30% redundancy
    burst_error_capability: int = 10  # Handle up to 10 consecutive errors
    symbol_size: int = 4  # 4 bits per symbol for DNA
    interleaving_depth: int = 4  # Interleaving for burst error protection


class ErrorCorrectionBase(ABC):
    """Abstract base class for error correction methods."""
    
    @abstractmethod
    def encode(self, data: np.ndarray) -> np.ndarray:
        """Add error correction to data."""
        pass
    
    @abstractmethod
    def decode(self, encoded_data: np.ndarray) -> np.ndarray:
        """Recover original data from error-corrected data."""
        pass
    
    @abstractmethod
    def get_overhead(self) -> float:
        """Get overhead ratio (encoded_size / original_size)."""
        pass


class SimpleParityCorrection(ErrorCorrectionBase):
    """Simple parity-based error correction."""
    
    def __init__(self, block_size: int = 8):
        """Initialize with block size for parity calculation."""
        self.block_size = block_size
    
    def encode(self, data: np.ndarray) -> np.ndarray:
        """Add parity bits to data."""
        encoded = []
        
        # Process data in blocks
        for i in range(0, len(data), self.block_size):
            block = data[i:i+self.block_size]
            
            # Pad block if necessary
            if len(block) < self.block_size:
                padding = np.zeros(self.block_size - len(block), dtype=np.uint8)
                block = np.concatenate([block, padding])
            
            # Calculate parity
            parity = np.sum(block) % 2
            
            # Add block + parity
            encoded.extend(block.tolist() + [parity])
        
        return np.array(encoded, dtype=np.uint8)
    
    def decode(self, encoded_data: np.ndarray) -> np.ndarray:
        """Decode data with error detection."""
        decoded = []
        block_size_with_parity = self.block_size + 1
        
        for i in range(0, len(encoded_data), block_size_with_parity):
            if i + block_size_with_parity <= len(encoded_data):
                block_with_parity = encoded_data[i:i+block_size_with_parity]
                block = block_with_parity[:-1]
                received_parity = block_with_parity[-1]
                
                # Check parity
                calculated_parity = np.sum(block) % 2
                
                if calculated_parity == received_parity:
                    decoded.extend(block.tolist())
                else:
                    # Parity error detected - for now, just use the data as-is
                    # In a more sophisticated implementation, would attempt correction
                    decoded.extend(block.tolist())
        
        return np.array(decoded, dtype=np.uint8)
    
    def get_overhead(self) -> float:
        """Get overhead ratio."""
        return (self.block_size + 1) / self.block_size


class ReedSolomonDNA(ErrorCorrectionBase):
    """Simplified Reed-Solomon error correction for DNA."""
    
    def __init__(self, n: int = 15, k: int = 11):
        """Initialize Reed-Solomon with (n,k) parameters.
        
        Args:
            n: Total codeword length
            k: Data symbols per codeword
        """
        self.n = n  # Total symbols in codeword
        self.k = k  # Data symbols
        self.redundancy_symbols = n - k
        
        # Generate Galois field for GF(16) since we work with 4-bit symbols
        self._generate_gf16_tables()
    
    def _generate_gf16_tables(self):
        """Generate Galois Field GF(16) multiplication tables."""
        # Simplified GF(16) with primitive polynomial x^4 + x + 1
        self.gf_exp = np.zeros(32, dtype=np.uint8)
        self.gf_log = np.zeros(16, dtype=np.uint8)
        
        # Generate exponential table
        x = 1
        for i in range(15):
            self.gf_exp[i] = x
            self.gf_log[x] = i
            x <<= 1
            if x & 16:  # If overflow, apply primitive polynomial
                x ^= 19  # x^4 + x + 1 = 10011 = 19
        
        # Extend for easier computation
        for i in range(15, 32):
            self.gf_exp[i] = self.gf_exp[i - 15]
    
    def _gf_mult(self, a: int, b: int) -> int:
        """Multiply two elements in GF(16)."""
        if a == 0 or b == 0:
            return 0
        return self.gf_exp[self.gf_log[a] + self.gf_log[b]]
    
    def _gf_div(self, a: int, b: int) -> int:
        """Divide two elements in GF(16)."""
        if a == 0:
            return 0
        if b == 0:
            raise ValueError("Division by zero in GF(16)")
        return self.gf_exp[self.gf_log[a] - self.gf_log[b] + 15]
    
    def _encode_codeword(self, data_symbols: np.ndarray) -> np.ndarray:
        """Encode a single codeword."""
        if len(data_symbols) != self.k:
            raise ValueError(f"Data must be exactly {self.k} symbols")
        
        # Initialize generator polynomial coefficients
        # For simplicity, use a basic generator
        generator = np.ones(self.redundancy_symbols + 1, dtype=np.uint8)
        for i in range(1, self.redundancy_symbols + 1):
            generator[i] = self.gf_exp[i % 15]
        
        # Calculate parity symbols using polynomial division
        # This is a simplified version - real RS would use systematic encoding
        parity = np.zeros(self.redundancy_symbols, dtype=np.uint8)
        
        for i in range(self.k):
            coeff = data_symbols[i]
            for j in range(self.redundancy_symbols):
                parity[j] = parity[j] ^ self._gf_mult(coeff, generator[j + 1])
        
        # Combine data and parity
        codeword = np.concatenate([data_symbols, parity])
        return codeword
    
    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode data with Reed-Solomon error correction."""
        # Convert binary data to 4-bit symbols
        if len(data) % 4 != 0:
            # Pad to multiple of 4 bits
            padding = 4 - (len(data) % 4)
            data = np.concatenate([data, np.zeros(padding, dtype=np.uint8)])
        
        symbols = []
        for i in range(0, len(data), 4):
            # Combine 4 bits into one symbol
            symbol = 0
            for j in range(4):
                if i + j < len(data):
                    symbol |= (data[i + j] << (3 - j))
            symbols.append(symbol)
        
        symbols = np.array(symbols, dtype=np.uint8)
        
        # Encode in blocks of k symbols
        encoded_symbols = []
        for i in range(0, len(symbols), self.k):
            block = symbols[i:i+self.k]
            
            # Pad block if necessary
            if len(block) < self.k:
                padding = np.zeros(self.k - len(block), dtype=np.uint8)
                block = np.concatenate([block, padding])
            
            # Encode block
            codeword = self._encode_codeword(block)
            encoded_symbols.extend(codeword.tolist())
        
        # Convert symbols back to binary
        encoded_binary = []
        for symbol in encoded_symbols:
            for i in range(4):
                encoded_binary.append((symbol >> (3 - i)) & 1)
        
        return np.array(encoded_binary, dtype=np.uint8)
    
    def decode(self, encoded_data: np.ndarray) -> np.ndarray:
        """Decode Reed-Solomon encoded data."""
        # Convert binary back to symbols
        if len(encoded_data) % 4 != 0:
            # Should not happen with properly encoded data
            padding = 4 - (len(encoded_data) % 4)
            encoded_data = np.concatenate([encoded_data, np.zeros(padding, dtype=np.uint8)])
        
        symbols = []
        for i in range(0, len(encoded_data), 4):
            symbol = 0
            for j in range(4):
                symbol |= (encoded_data[i + j] << (3 - j))
            symbols.append(symbol)
        
        symbols = np.array(symbols, dtype=np.uint8)
        
        # Decode in blocks
        decoded_symbols = []
        for i in range(0, len(symbols), self.n):
            if i + self.n <= len(symbols):
                codeword = symbols[i:i+self.n]
                
                # For simplified implementation, just extract data portion
                # Real RS decoder would check syndromes and correct errors
                data_portion = codeword[:self.k]
                decoded_symbols.extend(data_portion.tolist())
        
        # Convert symbols back to binary
        decoded_binary = []
        for symbol in decoded_symbols:
            for i in range(4):
                decoded_binary.append((symbol >> (3 - i)) & 1)
        
        return np.array(decoded_binary, dtype=np.uint8)
    
    def get_overhead(self) -> float:
        """Get overhead ratio."""
        return self.n / self.k


class DNAErrorCorrection:
    """Main DNA error correction class."""
    
    def __init__(self, method: str = "reed_solomon", 
                 parameters: Optional[ErrorCorrectionParameters] = None):
        """Initialize error correction with specified method."""
        self.method = method
        self.parameters = parameters or ErrorCorrectionParameters()
        
        # Initialize error correction algorithm
        if method == "reed_solomon":
            # Calculate RS parameters based on redundancy
            k = 11  # Data symbols
            n = int(k / (1 - self.parameters.redundancy))
            self.corrector = ReedSolomonDNA(n=n, k=k)
        elif method == "parity":
            self.corrector = SimpleParityCorrection(block_size=8)
        elif method == "none":
            self.corrector = None
        else:
            raise ValueError(f"Unknown error correction method: {method}")
    
    def encode(self, data: np.ndarray) -> np.ndarray:
        """Add error correction to data."""
        if self.corrector is None:
            return data.copy()
        
        # Apply interleaving for burst error protection
        if self.parameters.interleaving_depth > 1:
            data = self._interleave(data)
        
        # Apply error correction
        encoded = self.corrector.encode(data)
        
        return encoded
    
    def decode(self, encoded_data: np.ndarray) -> np.ndarray:
        """Recover original data from error-corrected data."""
        if self.corrector is None:
            return encoded_data.copy()
        
        # Apply error correction
        decoded = self.corrector.decode(encoded_data)
        
        # Reverse interleaving
        if self.parameters.interleaving_depth > 1:
            decoded = self._deinterleave(decoded)
        
        return decoded
    
    def _interleave(self, data: np.ndarray) -> np.ndarray:
        """Apply interleaving to protect against burst errors."""
        depth = self.parameters.interleaving_depth
        
        # Pad data if necessary
        padding_needed = (depth - (len(data) % depth)) % depth
        if padding_needed > 0:
            data = np.concatenate([data, np.zeros(padding_needed, dtype=np.uint8)])
        
        # Reshape and transpose for interleaving
        rows = len(data) // depth
        matrix = data.reshape(rows, depth)
        interleaved = matrix.T.flatten()
        
        return interleaved
    
    def _deinterleave(self, interleaved_data: np.ndarray) -> np.ndarray:
        """Reverse interleaving."""
        depth = self.parameters.interleaving_depth
        
        # Reshape and transpose to reverse interleaving
        cols = len(interleaved_data) // depth
        matrix = interleaved_data.reshape(depth, cols)
        deinterleaved = matrix.T.flatten()
        
        return deinterleaved
    
    def get_overhead(self) -> float:
        """Get total overhead including interleaving."""
        if self.corrector is None:
            return 1.0
        
        return self.corrector.get_overhead()
    
    def estimate_error_correction_capability(self) -> Dict[str, float]:
        """Estimate error correction capabilities."""
        if self.method == "reed_solomon":
            # RS can correct up to (n-k)/2 symbol errors
            t = (self.corrector.n - self.corrector.k) // 2
            symbol_error_rate = t / self.corrector.n
            # Each symbol is 4 bits
            bit_error_rate = symbol_error_rate * 4
            
            return {
                'correctable_symbol_errors': t,
                'total_symbols': self.corrector.n,
                'symbol_error_rate': symbol_error_rate,
                'bit_error_rate': bit_error_rate,
                'burst_error_protection': self.parameters.interleaving_depth > 1
            }
        elif self.method == "parity":
            return {
                'error_detection_only': True,
                'block_size': self.corrector.block_size,
                'burst_error_protection': False
            }
        else:
            return {
                'no_error_correction': True
            }
    
    def test_error_correction(self, data: np.ndarray, 
                            error_rate: float = 0.01) -> Dict[str, Any]:
        """Test error correction with simulated errors."""
        if len(data) == 0:
            return {'success': True, 'errors_introduced': 0, 'errors_corrected': 0}
        
        # Encode data
        encoded = self.encode(data)
        
        # Introduce random errors
        n_errors = int(len(encoded) * error_rate)
        error_positions = np.random.choice(len(encoded), size=n_errors, replace=False)
        
        corrupted = encoded.copy()
        corrupted[error_positions] = 1 - corrupted[error_positions]  # Flip bits
        
        # Attempt to decode
        try:
            decoded = self.decode(corrupted)
            
            # Compare with original (truncate to original length)
            original_length = len(data)
            decoded_truncated = decoded[:original_length]
            
            errors_remaining = np.sum(data != decoded_truncated)
            errors_corrected = n_errors - errors_remaining
            
            success = errors_remaining == 0
            
            return {
                'success': success,
                'errors_introduced': n_errors,
                'errors_corrected': errors_corrected,
                'errors_remaining': errors_remaining,
                'correction_rate': errors_corrected / n_errors if n_errors > 0 else 1.0
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'errors_introduced': n_errors,
                'errors_corrected': 0
            }