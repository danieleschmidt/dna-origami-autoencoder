"""Tests for data models."""

import pytest
import numpy as np
from dataclasses import FrozenInstanceError

from dna_origami_ae.models.dna_sequence import DNASequence, DNAConstraints
from dna_origami_ae.models.image_data import ImageData, ImageMetadata
from dna_origami_ae.models.origami_structure import OrigamiStructure, StapleStrand, ScaffoldPath
from dna_origami_ae.models.simulation_data import StructureCoordinates, TrajectoryData, SimulationResult, SimulationStatus


class TestDNASequence:
    """Test DNA sequence model."""
    
    def test_valid_dna_sequence(self):
        """Test creating valid DNA sequence."""
        seq = DNASequence(
            sequence="ATGCATGC",
            name="test_seq",
            description="Test sequence"
        )
        
        assert seq.sequence == "ATGCATGC"
        assert seq.name == "test_seq"
        assert len(seq) == 8
        assert seq.gc_content == 0.5  # 4 GC out of 8 bases
    
    def test_sequence_validation(self):
        """Test sequence validation."""
        # Valid sequence should work
        seq = DNASequence(sequence="ATGC")
        assert seq.sequence == "ATGC"
        
        # Invalid characters should raise error
        with pytest.raises(ValueError, match="Invalid DNA sequence"):
            DNASequence(sequence="ATGCX")
        
        # Empty sequence should raise error
        with pytest.raises(ValueError, match="Empty sequence"):
            DNASequence(sequence="")
    
    def test_sequence_properties(self):
        """Test calculated properties."""
        seq = DNASequence(sequence="AAATTTGGGCCC")
        
        # Test basic properties
        assert len(seq) == 12
        assert seq.gc_content == 0.5  # 6 GC out of 12
        
        # Test melting temperature calculation
        tm = seq.melting_temperature
        assert isinstance(tm, float)
        assert tm > 0
    
    def test_reverse_complement(self):
        """Test reverse complement calculation."""
        seq = DNASequence(sequence="ATGC")
        rev_comp = seq.reverse_complement()
        
        assert rev_comp == "GCAT"
    
    def test_constraints_application(self):
        """Test applying biological constraints."""
        constraints = DNAConstraints(
            gc_content_range=(0.4, 0.6),
            max_homopolymer_length=3
        )
        
        # Valid sequence
        seq = DNASequence(
            sequence="ATGCATGC",
            constraints=constraints
        )
        assert seq.constraints == constraints
        
        # Test constraint violation
        with pytest.raises(ValueError, match="GC content"):
            DNASequence(
                sequence="AAAAAAAAAA",  # GC content = 0
                constraints=constraints
            )
    
    def test_sequence_analysis(self):
        """Test detailed sequence analysis."""
        seq = DNASequence(sequence="ATGCATGCAAATTTGGGCCC")
        analysis = seq.analyze_sequence()
        
        assert 'length' in analysis
        assert 'gc_content' in analysis
        assert 'melting_temperature' in analysis
        assert 'homopolymer_runs' in analysis
        assert 'palindromes' in analysis
        
        assert analysis['length'] == 20
        assert 0 <= analysis['gc_content'] <= 1
    
    def test_sequence_immutability(self):
        """Test that sequence is immutable after creation."""
        seq = DNASequence(sequence="ATGC")
        
        # Should not be able to modify sequence
        with pytest.raises(AttributeError):
            seq.sequence = "CGTA"


class TestImageData:
    """Test image data model."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.test_array = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        self.test_image = ImageData.from_array(self.test_array, name="test")
    
    def test_from_array(self):
        """Test creating ImageData from numpy array."""
        image = ImageData.from_array(self.test_array, name="test")
        
        assert image.metadata.width == 32
        assert image.metadata.height == 32
        assert image.metadata.channels == 1
        assert image.metadata.bit_depth == 8
        assert image.name == "test"
        np.testing.assert_array_equal(image.data, self.test_array)
    
    def test_rgb_image(self):
        """Test RGB image creation."""
        rgb_array = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        image = ImageData.from_array(rgb_array, name="rgb_test")
        
        assert image.metadata.channels == 3
        assert image.metadata.width == 32
        assert image.metadata.height == 32
        assert image.data.shape == (32, 32, 3)
    
    def test_image_statistics(self):
        """Test image statistics calculation."""
        stats = self.test_image.get_statistics()
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'histogram' in stats
        
        assert stats['min'] >= 0
        assert stats['max'] <= 255
        assert len(stats['histogram']) == 256
    
    def test_image_preprocessing(self):
        """Test image preprocessing operations."""
        # Test normalization
        normalized = self.test_image.normalize()
        assert np.min(normalized.data) >= 0
        assert np.max(normalized.data) <= 1
        
        # Test resizing
        resized = self.test_image.resize((16, 16))
        assert resized.metadata.width == 16
        assert resized.metadata.height == 16
        
        # Test cropping
        cropped = self.test_image.crop((8, 8, 24, 24))
        assert cropped.metadata.width == 16
        assert cropped.metadata.height == 16
    
    def test_quality_metrics(self):
        """Test image quality metrics."""
        # Create similar image for comparison
        noisy_array = self.test_array + np.random.randint(-10, 10, self.test_array.shape)
        noisy_array = np.clip(noisy_array, 0, 255).astype(np.uint8)
        noisy_image = ImageData.from_array(noisy_array, name="noisy")
        
        # Test MSE
        mse = self.test_image.calculate_mse(noisy_image)
        assert mse >= 0
        
        # Test PSNR
        psnr = self.test_image.calculate_psnr(noisy_image)
        assert psnr > 0
        
        # Test SSIM
        ssim = self.test_image.calculate_ssim(noisy_image)
        assert 0 <= ssim <= 1
    
    def test_metadata_consistency(self):
        """Test metadata consistency."""
        metadata = self.test_image.metadata
        
        assert metadata.size_bytes == self.test_array.nbytes
        assert metadata.width * metadata.height == self.test_array.size
        
        # Test size calculation
        expected_size = 32 * 32 * 1 * 1  # width * height * channels * bytes_per_pixel
        assert metadata.size_bytes == expected_size
    
    def test_invalid_array_shapes(self):
        """Test handling of invalid array shapes."""
        # 1D array should raise error
        with pytest.raises(ValueError, match="Image array must be 2D or 3D"):
            ImageData.from_array(np.array([1, 2, 3]), name="invalid")
        
        # 4D array should raise error
        with pytest.raises(ValueError, match="Image array must be 2D or 3D"):
            ImageData.from_array(np.random.rand(10, 10, 3, 2), name="invalid")


class TestOrigamiStructure:
    """Test origami structure model."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.scaffold_seq = DNASequence(
            sequence="ATGCATGCATGCATGC" * 10,  # Longer sequence
            name="scaffold"
        )
        
        self.scaffold_path = ScaffoldPath(
            sequence=self.scaffold_seq,
            path_coordinates=[(0, i, i) for i in range(len(self.scaffold_seq))],
            routing_method="honeycomb"
        )
        
        # Create some staples
        self.staples = [
            StapleStrand(
                sequence=DNASequence(sequence="ATGCATGCATGC", name="staple_0"),
                start_helix=0,
                start_position=0,
                end_helix=0,
                end_position=11,
                crossovers=[],
                color="#FF0000"
            ),
            StapleStrand(
                sequence=DNASequence(sequence="CGTAGCTAGCTA", name="staple_1"),
                start_helix=0,
                start_position=12,
                end_helix=0,
                end_position=23,
                crossovers=[],
                color="#00FF00"
            )
        ]
    
    def test_origami_structure_creation(self):
        """Test creating origami structure."""
        structure = OrigamiStructure(
            name="test_origami",
            scaffold=self.scaffold_path,
            staples=self.staples,
            target_shape="square",
            dimensions=(100.0, 100.0, 10.0)
        )
        
        assert structure.name == "test_origami"
        assert structure.scaffold == self.scaffold_path
        assert len(structure.staples) == 2
        assert structure.target_shape == "square"
    
    def test_structure_validation(self):
        """Test structure validation."""
        structure = OrigamiStructure(
            name="test_origami",
            scaffold=self.scaffold_path,
            staples=self.staples,
            target_shape="square",
            dimensions=(100.0, 100.0, 10.0)
        )
        
        is_valid, errors = structure.validate_design()
        
        # Should be valid with our test setup
        assert is_valid
        assert len(errors) == 0
    
    def test_structure_statistics(self):
        """Test structure statistics."""
        structure = OrigamiStructure(
            name="test_origami",
            scaffold=self.scaffold_path,
            staples=self.staples,
            target_shape="square",
            dimensions=(100.0, 100.0, 10.0)
        )
        
        stats = structure.get_design_statistics()
        
        assert 'scaffold_length' in stats
        assert 'num_staples' in stats
        assert 'total_staple_length' in stats
        assert 'average_staple_length' in stats
        
        assert stats['scaffold_length'] == len(self.scaffold_seq)
        assert stats['num_staples'] == 2
    
    def test_sequence_export(self):
        """Test sequence export functionality."""
        structure = OrigamiStructure(
            name="test_origami",
            scaffold=self.scaffold_path,
            staples=self.staples,
            target_shape="square",
            dimensions=(100.0, 100.0, 10.0)
        )
        
        # Test getting all sequences
        all_sequences = structure.get_all_sequences()
        
        assert len(all_sequences) == 3  # 1 scaffold + 2 staples
        assert all_sequences[0] == self.scaffold_seq
        assert all_sequences[1] == self.staples[0].sequence
        assert all_sequences[2] == self.staples[1].sequence
    
    def test_staple_optimization(self):
        """Test staple optimization."""
        structure = OrigamiStructure(
            name="test_origami",
            scaffold=self.scaffold_path,
            staples=self.staples,
            target_shape="square",
            dimensions=(100.0, 100.0, 10.0)
        )
        
        # Test optimization
        structure.optimize_staple_lengths(target_length=16, tolerance=4)
        
        # Staples should still be valid after optimization
        for staple in structure.staples:
            assert isinstance(staple.sequence, DNASequence)


class TestSimulationData:
    """Test simulation data models."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Create test coordinates
        self.positions = np.random.normal(0, 10, (100, 3))
        self.atom_types = ['C'] * 100
        self.coords = StructureCoordinates(
            positions=self.positions,
            atom_types=self.atom_types
        )
    
    def test_structure_coordinates(self):
        """Test StructureCoordinates model."""
        assert self.coords.n_atoms == 100
        assert self.coords.positions.shape == (100, 3)
        assert len(self.coords.atom_types) == 100
        
        # Test center of mass
        com = self.coords.center_of_mass
        assert com.shape == (3,)
        
        # Test radius of gyration
        rg = self.coords.radius_of_gyration
        assert rg > 0
    
    def test_trajectory_data(self):
        """Test TrajectoryData model."""
        # Create multiple frames
        frames = []
        timestamps = []
        
        for i in range(10):
            positions = self.positions + np.random.normal(0, 0.1, self.positions.shape)
            frame = StructureCoordinates(
                positions=positions,
                atom_types=self.atom_types.copy()
            )
            frames.append(frame)
            timestamps.append(i * 0.1)
        
        trajectory = TrajectoryData(
            frames=frames,
            timestamps=np.array(timestamps),
            temperature=300.0
        )
        
        assert trajectory.n_frames == 10
        assert trajectory.simulation_time == 0.9
        assert trajectory.time_step == 0.1
        
        # Test RMSD calculation
        rmsd_series = trajectory.calculate_rmsd()
        assert len(rmsd_series) == 10
        assert rmsd_series[0] == 0.0  # RMSD with self
    
    def test_simulation_result(self):
        """Test SimulationResult model."""
        # Create simple trajectory
        frames = [self.coords]
        trajectory = TrajectoryData(
            frames=frames,
            timestamps=np.array([0.0]),
            temperature=300.0
        )
        
        result = SimulationResult(
            trajectory=trajectory,
            status=SimulationStatus.COMPLETED,
            computation_time=120.0
        )
        
        assert result.success
        assert result.final_structure == self.coords
        assert result.computation_time == 120.0
        
        # Test quality metrics
        metrics = result.get_quality_metrics()
        assert 'quality_score' in metrics
        assert 0 <= metrics['quality_score'] <= 1
    
    def test_coordinate_validation(self):
        """Test coordinate validation."""
        # Valid coordinates should work
        coords = StructureCoordinates(
            positions=np.array([[0, 0, 0], [1, 1, 1]]),
            atom_types=['C', 'N']
        )
        assert coords.n_atoms == 2
        
        # Mismatched atom types should raise error
        with pytest.raises(ValueError, match="Number of atom types"):
            StructureCoordinates(
                positions=np.array([[0, 0, 0], [1, 1, 1]]),
                atom_types=['C']  # Wrong number
            )
        
        # Wrong position dimensions should raise error
        with pytest.raises(ValueError, match="Positions must have shape"):
            StructureCoordinates(
                positions=np.array([[0, 0], [1, 1]]),  # 2D instead of 3D
                atom_types=['C', 'N']
            )


class TestDataModelIntegration:
    """Integration tests for data models."""
    
    def test_complete_origami_workflow(self):
        """Test complete workflow with all models."""
        # Create image
        image_array = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
        image = ImageData.from_array(image_array, name="integration_test")
        
        # Create DNA sequences (simulating encoding result)
        sequences = [
            DNASequence(
                sequence="ATGCATGC" * 10,
                name="encoded_chunk_0"
            )
        ]
        
        # Create origami structure
        scaffold_path = ScaffoldPath(
            sequence=sequences[0],
            path_coordinates=[(0, i, i) for i in range(len(sequences[0]))],
            routing_method="honeycomb"
        )
        
        staples = [
            StapleStrand(
                sequence=DNASequence(sequence="ATGCATGC", name="staple_0"),
                start_helix=0,
                start_position=0,
                end_helix=0,
                end_position=7,
                crossovers=[],
                color="#FF0000"
            )
        ]
        
        structure = OrigamiStructure(
            name="integration_structure",
            scaffold=scaffold_path,
            staples=staples,
            target_shape="square",
            dimensions=(50.0, 50.0, 5.0)
        )
        
        # Validate structure
        is_valid, errors = structure.validate_design()
        assert is_valid
        
        # Create simulation coordinates
        positions = np.random.normal(0, 5, (len(sequences[0]) + len(staples[0].sequence), 3))
        atom_types = ['C'] * len(positions)
        
        coords = StructureCoordinates(
            positions=positions,
            atom_types=atom_types
        )
        
        # Create simulation result
        trajectory = TrajectoryData(
            frames=[coords],
            timestamps=np.array([0.0]),
            temperature=300.0
        )
        
        result = SimulationResult(
            trajectory=trajectory,
            status=SimulationStatus.COMPLETED,
            metadata={
                'original_image': image.name,
                'structure_name': structure.name
            }
        )
        
        assert result.success
        assert result.metadata['original_image'] == image.name
        assert result.metadata['structure_name'] == structure.name