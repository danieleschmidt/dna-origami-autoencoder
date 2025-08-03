"""DNA-compatible compression algorithms."""

import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class CompressionParameters:
    """Parameters for DNA compression algorithms."""
    
    method: str = "rle"  # Run-length encoding
    block_size: int = 64  # Block size for block-based methods
    dictionary_size: int = 256  # Dictionary size for LZ methods
    preserve_biological_constraints: bool = True


class CompressionBase(ABC):
    """Abstract base class for compression methods."""
    
    @abstractmethod
    def compress(self, data: np.ndarray) -> np.ndarray:
        """Compress binary data."""
        pass
    
    @abstractmethod
    def decompress(self, compressed_data: np.ndarray) -> np.ndarray:
        """Decompress data."""
        pass
    
    @abstractmethod
    def get_compression_ratio(self, original_size: int, compressed_size: int) -> float:
        """Calculate compression ratio."""
        return original_size / compressed_size if compressed_size > 0 else 0.0


class RunLengthEncoding(CompressionBase):
    """Run-length encoding optimized for DNA storage."""
    
    def __init__(self, max_run_length: int = 63):
        """Initialize with maximum run length (6 bits = 63 max)."""
        self.max_run_length = max_run_length
    
    def compress(self, data: np.ndarray) -> np.ndarray:
        """Compress using run-length encoding."""
        if len(data) == 0:
            return np.array([], dtype=np.uint8)
        
        compressed = []
        i = 0
        
        while i < len(data):
            current_bit = data[i]
            run_length = 1
            
            # Count consecutive identical bits
            while (i + run_length < len(data) and 
                   data[i + run_length] == current_bit and 
                   run_length < self.max_run_length):
                run_length += 1
            
            # Encode run: 6 bits for length + 1 bit for value
            length_bits = [(run_length >> j) & 1 for j in range(6)]
            compressed.extend(length_bits + [current_bit])
            
            i += run_length
        
        return np.array(compressed, dtype=np.uint8)
    
    def decompress(self, compressed_data: np.ndarray) -> np.ndarray:
        """Decompress run-length encoded data."""
        if len(compressed_data) == 0:
            return np.array([], dtype=np.uint8)
        
        if len(compressed_data) % 7 != 0:
            raise ValueError("Compressed data length must be multiple of 7")
        
        decompressed = []
        
        for i in range(0, len(compressed_data), 7):
            # Extract 6 bits for length + 1 bit for value
            length_bits = compressed_data[i:i+6]
            bit_value = compressed_data[i+6]
            
            # Reconstruct run length
            run_length = sum(length_bits[j] * (2 ** j) for j in range(6))
            
            # Add repeated bits
            decompressed.extend([bit_value] * run_length)
        
        return np.array(decompressed, dtype=np.uint8)


class LempelZivDNA(CompressionBase):
    """Modified Lempel-Ziv compression for DNA sequences."""
    
    def __init__(self, dictionary_size: int = 256, min_match_length: int = 3):
        """Initialize LZ compressor."""
        self.dictionary_size = dictionary_size
        self.min_match_length = min_match_length
    
    def compress(self, data: np.ndarray) -> np.ndarray:
        """Compress using LZ77-style algorithm."""
        if len(data) == 0:
            return np.array([], dtype=np.uint8)
        
        compressed = []
        dictionary = {}
        i = 0
        
        while i < len(data):
            # Find longest match in dictionary
            best_match = None
            best_length = 0
            
            for length in range(self.min_match_length, 
                              min(len(data) - i + 1, self.dictionary_size)):
                pattern = tuple(data[i:i+length])
                
                if pattern in dictionary and length > best_length:
                    best_match = dictionary[pattern]
                    best_length = length
            
            if best_match is not None and best_length >= self.min_match_length:
                # Encode as reference: (distance, length)
                distance = i - best_match
                
                # Encode distance (8 bits) + length (8 bits) + flag (1 bit = 1)
                distance_bits = [(distance >> j) & 1 for j in range(8)]
                length_bits = [(best_length >> j) & 1 for j in range(8)]
                compressed.extend([1] + distance_bits + length_bits)
                
                # Add current pattern to dictionary
                pattern = tuple(data[i:i+best_length])
                dictionary[pattern] = i
                
                i += best_length
            else:
                # Literal byte: flag (1 bit = 0) + data (8 bits)
                if i + 8 <= len(data):
                    byte_bits = data[i:i+8].tolist()
                    compressed.extend([0] + byte_bits)
                    i += 8
                else:
                    # Handle remaining bits
                    remaining = data[i:].tolist()
                    padded = remaining + [0] * (8 - len(remaining))
                    compressed.extend([0] + padded)
                    i = len(data)
        
        return np.array(compressed, dtype=np.uint8)
    
    def decompress(self, compressed_data: np.ndarray) -> np.ndarray:
        """Decompress LZ-compressed data."""
        if len(compressed_data) == 0:
            return np.array([], dtype=np.uint8)
        
        decompressed = []
        i = 0
        
        while i < len(compressed_data):
            if i >= len(compressed_data):
                break
                
            flag = compressed_data[i]
            i += 1
            
            if flag == 0:  # Literal
                if i + 8 <= len(compressed_data):
                    literal_bits = compressed_data[i:i+8]
                    decompressed.extend(literal_bits.tolist())
                    i += 8
                else:
                    break
            else:  # Reference
                if i + 16 <= len(compressed_data):
                    distance_bits = compressed_data[i:i+8]
                    length_bits = compressed_data[i+8:i+16]
                    
                    distance = sum(distance_bits[j] * (2 ** j) for j in range(8))
                    length = sum(length_bits[j] * (2 ** j) for j in range(8))
                    
                    # Copy from previous data
                    start_pos = len(decompressed) - distance
                    for j in range(length):
                        if start_pos + j < len(decompressed):
                            decompressed.append(decompressed[start_pos + j])
                    
                    i += 16
                else:
                    break
        
        return np.array(decompressed, dtype=np.uint8)


class AdaptiveHuffmanDNA(CompressionBase):
    """Adaptive Huffman coding for DNA sequences."""
    
    def __init__(self):
        """Initialize adaptive Huffman coder."""
        self.symbol_frequencies = {}
        self.huffman_tree = None
        self.codes = {}
    
    def _build_huffman_tree(self, frequencies: Dict[int, int]) -> Dict[int, str]:
        """Build Huffman tree and return codes."""
        if not frequencies:
            return {}
        
        if len(frequencies) == 1:
            # Special case: only one symbol
            symbol = list(frequencies.keys())[0]
            return {symbol: '0'}
        
        # Create priority queue (simple implementation)
        heap = [(freq, [symbol]) for symbol, freq in frequencies.items()]
        heap.sort(key=lambda x: x[0])
        
        # Build tree
        while len(heap) > 1:
            freq1, symbols1 = heap.pop(0)
            freq2, symbols2 = heap.pop(0)
            
            merged_freq = freq1 + freq2
            merged_symbols = symbols1 + symbols2
            
            # Insert back in sorted order
            inserted = False
            for i, (freq, _) in enumerate(heap):
                if merged_freq <= freq:
                    heap.insert(i, (merged_freq, merged_symbols))
                    inserted = True
                    break
            
            if not inserted:
                heap.append((merged_freq, merged_symbols))
        
        # Generate codes
        codes = {}
        if heap:
            self._generate_codes(heap[0][1], '', codes)
        
        return codes
    
    def _generate_codes(self, symbols: List[int], prefix: str, codes: Dict[int, str]):
        """Recursively generate Huffman codes."""
        if len(symbols) == 1:
            codes[symbols[0]] = prefix or '0'
        else:
            mid = len(symbols) // 2
            self._generate_codes(symbols[:mid], prefix + '0', codes)
            self._generate_codes(symbols[mid:], prefix + '1', codes)
    
    def compress(self, data: np.ndarray) -> np.ndarray:
        """Compress using adaptive Huffman coding."""
        if len(data) == 0:
            return np.array([], dtype=np.uint8)
        
        # Count symbol frequencies
        frequencies = {}
        for symbol in data:
            frequencies[symbol] = frequencies.get(symbol, 0) + 1
        
        # Build Huffman codes
        codes = self._build_huffman_tree(frequencies)
        
        # Encode header (frequencies)
        header = self._encode_header(frequencies)
        
        # Encode data
        encoded_bits = []
        for symbol in data:
            if symbol in codes:
                encoded_bits.extend([int(bit) for bit in codes[symbol]])
        
        # Combine header and data
        compressed = header + encoded_bits
        
        # Pad to byte boundary
        while len(compressed) % 8 != 0:
            compressed.append(0)
        
        return np.array(compressed, dtype=np.uint8)
    
    def _encode_header(self, frequencies: Dict[int, int]) -> List[int]:
        """Encode frequency table in header."""
        header = []
        
        # Number of symbols (8 bits)
        num_symbols = len(frequencies)
        header.extend([(num_symbols >> i) & 1 for i in range(8)])
        
        # Symbol frequencies
        for symbol, freq in frequencies.items():
            # Symbol value (8 bits) + frequency (16 bits)
            header.extend([(symbol >> i) & 1 for i in range(8)])
            header.extend([(freq >> i) & 1 for i in range(16)])
        
        return header
    
    def decompress(self, compressed_data: np.ndarray) -> np.ndarray:
        """Decompress Huffman-coded data."""
        if len(compressed_data) == 0:
            return np.array([], dtype=np.uint8)
        
        # Decode header
        frequencies, header_length = self._decode_header(compressed_data)
        
        # Rebuild Huffman codes
        codes = self._build_huffman_tree(frequencies)
        
        # Create reverse lookup
        reverse_codes = {code: symbol for symbol, code in codes.items()}
        
        # Decode data
        data_bits = compressed_data[header_length:]
        decompressed = []
        
        current_code = ''
        for bit in data_bits:
            current_code += str(bit)
            
            if current_code in reverse_codes:
                decompressed.append(reverse_codes[current_code])
                current_code = ''
        
        return np.array(decompressed, dtype=np.uint8)
    
    def _decode_header(self, data: np.ndarray) -> tuple:
        """Decode frequency table from header."""
        if len(data) < 8:
            return {}, 0
        
        # Number of symbols
        num_symbols = sum(data[i] * (2 ** i) for i in range(8))
        
        frequencies = {}
        pos = 8
        
        for _ in range(num_symbols):
            if pos + 24 <= len(data):
                # Symbol (8 bits) + frequency (16 bits)
                symbol = sum(data[pos + i] * (2 ** i) for i in range(8))
                freq = sum(data[pos + 8 + i] * (2 ** i) for i in range(16))
                
                frequencies[symbol] = freq
                pos += 24
            else:
                break
        
        return frequencies, pos


class DNACompression:
    """Main DNA compression class with multiple algorithms."""
    
    def __init__(self, method: str = "rle", 
                 parameters: Optional[CompressionParameters] = None):
        """Initialize with specified compression method."""
        self.method = method
        self.parameters = parameters or CompressionParameters()
        
        # Initialize compressor
        if method == "rle":
            self.compressor = RunLengthEncoding()
        elif method == "lz":
            self.compressor = LempelZivDNA(
                dictionary_size=self.parameters.dictionary_size
            )
        elif method == "huffman":
            self.compressor = AdaptiveHuffmanDNA()
        elif method == "none":
            self.compressor = None
        else:
            raise ValueError(f"Unknown compression method: {method}")
    
    def compress(self, data: np.ndarray) -> np.ndarray:
        """Compress data using selected method."""
        if self.compressor is None:
            return data.copy()
        
        return self.compressor.compress(data)
    
    def decompress(self, compressed_data: np.ndarray) -> np.ndarray:
        """Decompress data."""
        if self.compressor is None:
            return compressed_data.copy()
        
        return self.compressor.decompress(compressed_data)
    
    def analyze_compression(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze compression performance."""
        if self.compressor is None:
            return {
                'method': 'none',
                'original_size': len(data),
                'compressed_size': len(data),
                'compression_ratio': 1.0,
                'space_savings': 0.0
            }
        
        compressed = self.compress(data)
        compression_ratio = self.compressor.get_compression_ratio(len(data), len(compressed))
        space_savings = 1.0 - (len(compressed) / len(data)) if len(data) > 0 else 0.0
        
        return {
            'method': self.method,
            'original_size': len(data),
            'compressed_size': len(compressed),
            'compression_ratio': compression_ratio,
            'space_savings': space_savings,
            'effective_bits_per_symbol': len(compressed) / len(data) if len(data) > 0 else 0
        }
    
    def benchmark_methods(self, data: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Benchmark different compression methods."""
        methods = ['rle', 'lz', 'huffman', 'none']
        results = {}
        
        for method in methods:
            try:
                compressor = DNACompression(method)
                analysis = compressor.analyze_compression(data)
                
                # Test round-trip
                compressed = compressor.compress(data)
                decompressed = compressor.decompress(compressed)
                
                round_trip_success = np.array_equal(data, decompressed)
                
                results[method] = {
                    **analysis,
                    'round_trip_success': round_trip_success
                }
                
            except Exception as e:
                results[method] = {
                    'error': str(e),
                    'round_trip_success': False
                }
        
        return results