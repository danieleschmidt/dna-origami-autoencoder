"""Complete end-to-end pipeline example for DNA origami autoencoder."""

import numpy as np
from typing import Dict, Any, List, Optional
import time

from ..dna_origami_ae.models.image_data import ImageData
from ..dna_origami_ae.models.origami_structure import OrigamiStructure
from ..dna_origami_ae.models.simulation_data import StructureCoordinates, TrajectoryData, SimulationResult, SimulationStatus
from ..dna_origami_ae.encoding.image_encoder import DNAEncoder, EncodingParameters
from ..dna_origami_ae.design.origami_designer import OrigamiDesigner, DesignParameters
from ..dna_origami_ae.simulation.origami_simulator import OrigamiSimulator
from ..dna_origami_ae.decoding.transformer_decoder import TransformerDecoder
from ..dna_origami_ae.wetlab.protocol_generator import ProtocolGenerator, LabEquipment
from ..dna_origami_ae.wetlab.imaging_processor import AFMProcessor


class CompletePipelineExample:
    """Demonstrate complete DNA origami autoencoder pipeline."""
    
    def __init__(self):
        """Initialize pipeline components."""
        self.encoder = DNAEncoder(bits_per_base=2, error_correction='reed_solomon')
        self.designer = OrigamiDesigner()
        self.simulator = OrigamiSimulator()
        self.decoder = TransformerDecoder()
        self.protocol_generator = ProtocolGenerator(equipment=LabEquipment.STANDARD_BIO_LAB)
        self.afm_processor = AFMProcessor()
        
        # Pipeline statistics
        self.stats = {
            'total_runs': 0,
            'successful_runs': 0,
            'average_reconstruction_accuracy': 0.0,
            'average_pipeline_time': 0.0
        }
    
    def run_complete_pipeline(self, 
                            input_image: Optional[ImageData] = None,
                            simulate_folding: bool = True,
                            generate_protocol: bool = True,
                            validate_reconstruction: bool = True) -> Dict[str, Any]:
        """Run complete pipeline from image to reconstruction."""
        print("🚀 Starting complete DNA origami autoencoder pipeline...")
        start_time = time.time()
        
        results = {
            'success': False,
            'pipeline_steps': {},
            'metrics': {},
            'artifacts': {}
        }
        
        try:
            # Step 1: Generate or use input image
            if input_image is None:
                input_image = self._create_sample_image()
            
            print(f"📸 Input image: {input_image}")
            results['artifacts']['input_image'] = input_image
            
            # Step 2: DNA Encoding
            print("🧬 Encoding image to DNA sequences...")
            encoding_start = time.time()
            
            encoding_params = EncodingParameters(
                bits_per_base=2,
                error_correction="reed_solomon",
                redundancy=0.3,
                chunk_size=200,
                include_metadata=True
            )
            
            dna_sequences = self.encoder.encode_image(input_image, encoding_params)
            encoding_time = time.time() - encoding_start
            
            print(f"   Generated {len(dna_sequences)} DNA sequences")
            print(f"   Total bases: {sum(len(seq) for seq in dna_sequences)}")
            print(f"   Encoding time: {encoding_time:.2f}s")
            
            results['pipeline_steps']['encoding'] = {
                'sequences_generated': len(dna_sequences),
                'total_bases': sum(len(seq) for seq in dna_sequences),
                'encoding_time': encoding_time
            }
            results['artifacts']['dna_sequences'] = dna_sequences
            
            # Step 3: Origami Design
            print("🏗️  Designing DNA origami structure...")
            design_start = time.time()
            
            design_params = DesignParameters(
                scaffold_length=7249,
                staple_length=32,
                target_shape="square",
                dimensions=(100.0, 100.0),
                routing_method="honeycomb"
            )
            self.designer.params = design_params
            
            # Use first DNA sequence as scaffold (simplified)
            scaffold_sequence = dna_sequences[0] if dna_sequences else None
            origami_structure = self.designer.design_origami(
                scaffold_sequence, 
                target_shape="square",
                dimensions=(100.0, 100.0)
            )
            
            design_time = time.time() - design_start
            print(f"   Created structure with {len(origami_structure.staples)} staples")
            print(f"   Design time: {design_time:.2f}s")
            
            results['pipeline_steps']['design'] = {
                'staple_count': len(origami_structure.staples),
                'design_time': design_time,
                'target_shape': design_params.target_shape
            }
            results['artifacts']['origami_structure'] = origami_structure
            
            # Step 4: Molecular Simulation (if requested)
            simulation_result = None
            if simulate_folding:
                print("⚗️  Running molecular dynamics simulation...")
                simulation_start = time.time()
                
                simulation_result = self.simulator.simulate_folding(
                    origami_structure,
                    time_steps=100000,
                    temperature=300.0,
                    save_trajectory=True
                )
                
                simulation_time = time.time() - simulation_start
                print(f"   Simulation completed: {simulation_result.status.value}")
                print(f"   Simulation time: {simulation_time:.2f}s")
                
                if simulation_result.success:
                    quality_metrics = simulation_result.get_quality_metrics()
                    print(f"   Quality score: {quality_metrics['quality_score']:.3f}")
                
                results['pipeline_steps']['simulation'] = {
                    'status': simulation_result.status.value,
                    'simulation_time': simulation_time,
                    'quality_metrics': simulation_result.get_quality_metrics() if simulation_result.success else {}
                }
                results['artifacts']['simulation_result'] = simulation_result
            
            # Step 5: Neural Decoding
            print("🧠 Decoding structure with transformer model...")
            decoding_start = time.time()
            
            # Use simulated structure if available, otherwise create synthetic coordinates
            if simulation_result and simulation_result.success:
                final_structure_coords = simulation_result.final_structure
            else:
                # Create synthetic structure coordinates for demonstration
                final_structure_coords = self._create_synthetic_structure_coordinates()
            
            reconstructed_image = self.decoder.decode_structure(final_structure_coords)
            decoding_time = time.time() - decoding_start
            
            print(f"   Reconstructed image: {reconstructed_image}")
            print(f"   Decoding time: {decoding_time:.2f}s")
            
            results['pipeline_steps']['decoding'] = {
                'decoding_time': decoding_time,
                'output_shape': reconstructed_image.data.shape
            }
            results['artifacts']['reconstructed_image'] = reconstructed_image
            
            # Step 6: Quality Assessment
            if validate_reconstruction:
                print("📊 Validating reconstruction quality...")
                
                mse = input_image.calculate_mse(reconstructed_image)
                psnr = input_image.calculate_psnr(reconstructed_image)
                ssim = input_image.calculate_ssim(reconstructed_image)
                
                print(f"   MSE: {mse:.4f}")
                print(f"   PSNR: {psnr:.2f} dB")
                print(f"   SSIM: {ssim:.4f}")
                
                results['metrics'] = {
                    'mse': mse,
                    'psnr': psnr,
                    'ssim': ssim,
                    'reconstruction_accuracy': 1.0 / (1.0 + mse)  # Simple accuracy metric
                }
            
            # Step 7: Wet-lab Protocol Generation (if requested)
            if generate_protocol:
                print("🔬 Generating wet-lab assembly protocol...")
                protocol = self.protocol_generator.generate_protocol(
                    origami_structure,
                    scale="single_tube",
                    replicates=3
                )
                
                print(f"   Generated protocol with {len(protocol.steps)} steps")
                print(f"   Estimated time: {protocol.estimated_time_hours:.1f} hours")
                
                results['artifacts']['lab_protocol'] = protocol
            
            # Pipeline completed successfully
            total_time = time.time() - start_time
            print(f"✅ Pipeline completed successfully in {total_time:.2f}s")
            
            results['success'] = True
            results['total_time'] = total_time
            
            # Update statistics
            self._update_pipeline_stats(total_time, True, results.get('metrics', {}).get('reconstruction_accuracy', 0.0))
            
            return results
            
        except Exception as e:
            total_time = time.time() - start_time
            print(f"❌ Pipeline failed after {total_time:.2f}s: {e}")
            
            results['success'] = False
            results['error'] = str(e)
            results['total_time'] = total_time
            
            self._update_pipeline_stats(total_time, False, 0.0)
            
            return results
    
    def _create_sample_image(self) -> ImageData:
        """Create a sample image for demonstration."""
        # Create a simple pattern - DNA origami shape
        image_data = np.zeros((32, 32), dtype=np.uint8)
        
        # Create a square pattern with some features
        image_data[8:24, 8:24] = 200  # Main square
        image_data[12:20, 12:20] = 100  # Inner square
        image_data[14:18, 14:18] = 255  # Center bright spot
        
        # Add some noise for realism
        noise = np.random.normal(0, 10, (32, 32))
        image_data = np.clip(image_data.astype(float) + noise, 0, 255).astype(np.uint8)
        
        return ImageData.from_array(image_data, name="sample_origami_pattern")
    
    def _create_synthetic_structure_coordinates(self) -> StructureCoordinates:
        """Create synthetic structure coordinates for testing."""
        # Generate coordinates for a square-like DNA origami structure
        n_points = 1000
        
        # Create square perimeter points
        side_length = 100.0  # nm
        points_per_side = n_points // 4
        
        coords = []
        
        # Top side
        for i in range(points_per_side):
            x = (i / points_per_side) * side_length
            y = 0.0
            z = np.random.normal(0, 1.0)  # Small height variation
            coords.append([x, y, z])
        
        # Right side
        for i in range(points_per_side):
            x = side_length
            y = (i / points_per_side) * side_length
            z = np.random.normal(0, 1.0)
            coords.append([x, y, z])
        
        # Bottom side
        for i in range(points_per_side):
            x = side_length - (i / points_per_side) * side_length
            y = side_length
            z = np.random.normal(0, 1.0)
            coords.append([x, y, z])
        
        # Left side
        for i in range(points_per_side):
            x = 0.0
            y = side_length - (i / points_per_side) * side_length
            z = np.random.normal(0, 1.0)
            coords.append([x, y, z])
        
        # Fill remaining points with interior
        remaining = n_points - len(coords)
        for _ in range(remaining):
            x = np.random.uniform(10, side_length - 10)
            y = np.random.uniform(10, side_length - 10)
            z = np.random.normal(0, 0.5)
            coords.append([x, y, z])
        
        positions = np.array(coords)
        
        return StructureCoordinates(
            positions=positions,
            structure_type="synthetic_origami",
            coordinate_system="cartesian",
            units="nanometers"
        )
    
    def _update_pipeline_stats(self, pipeline_time: float, success: bool, accuracy: float) -> None:
        """Update pipeline statistics."""
        self.stats['total_runs'] += 1
        
        if success:
            self.stats['successful_runs'] += 1
            
            # Update average accuracy
            prev_avg_acc = self.stats['average_reconstruction_accuracy']
            success_count = self.stats['successful_runs']
            self.stats['average_reconstruction_accuracy'] = (
                (prev_avg_acc * (success_count - 1) + accuracy) / success_count
            )
        
        # Update average pipeline time
        prev_avg_time = self.stats['average_pipeline_time']
        total_count = self.stats['total_runs']
        self.stats['average_pipeline_time'] = (
            (prev_avg_time * (total_count - 1) + pipeline_time) / total_count
        )
    
    def run_batch_experiment(self, 
                           batch_size: int = 5,
                           image_variations: List[str] = None) -> Dict[str, Any]:
        """Run batch experiment with multiple images."""
        if image_variations is None:
            image_variations = ['square', 'circle', 'triangle', 'hexagon', 'cross']
        
        print(f"🔬 Running batch experiment with {batch_size} images...")
        
        batch_results = {
            'batch_size': batch_size,
            'individual_results': [],
            'aggregate_metrics': {},
            'success_rate': 0.0
        }
        
        successful_runs = 0
        total_metrics = {'mse': [], 'psnr': [], 'ssim': [], 'pipeline_time': []}
        
        for i in range(batch_size):
            shape = image_variations[i % len(image_variations)]
            print(f"\n--- Batch Run {i+1}/{batch_size}: {shape} ---")
            
            # Create specific test image
            test_image = self._create_test_image_by_shape(shape)
            
            # Run pipeline
            result = self.run_complete_pipeline(
                input_image=test_image,
                simulate_folding=True,
                generate_protocol=False,  # Skip for batch to save time
                validate_reconstruction=True
            )
            
            batch_results['individual_results'].append({
                'run_id': i + 1,
                'shape': shape,
                'success': result['success'],
                'metrics': result.get('metrics', {}),
                'pipeline_time': result['total_time']
            })
            
            if result['success']:
                successful_runs += 1
                metrics = result['metrics']
                total_metrics['mse'].append(metrics['mse'])
                total_metrics['psnr'].append(metrics['psnr'])
                total_metrics['ssim'].append(metrics['ssim'])
            
            total_metrics['pipeline_time'].append(result['total_time'])
        
        # Calculate aggregate metrics
        batch_results['success_rate'] = successful_runs / batch_size
        
        if successful_runs > 0:
            batch_results['aggregate_metrics'] = {
                'avg_mse': np.mean(total_metrics['mse']),
                'avg_psnr': np.mean(total_metrics['psnr']),
                'avg_ssim': np.mean(total_metrics['ssim']),
                'std_mse': np.std(total_metrics['mse']),
                'std_psnr': np.std(total_metrics['psnr']),
                'std_ssim': np.std(total_metrics['ssim'])
            }
        
        batch_results['avg_pipeline_time'] = np.mean(total_metrics['pipeline_time'])
        
        print(f"\n📊 Batch Experiment Complete!")
        print(f"   Success Rate: {batch_results['success_rate']:.1%}")
        print(f"   Average Pipeline Time: {batch_results['avg_pipeline_time']:.2f}s")
        if successful_runs > 0:
            print(f"   Average PSNR: {batch_results['aggregate_metrics']['avg_psnr']:.2f} dB")
            print(f"   Average SSIM: {batch_results['aggregate_metrics']['avg_ssim']:.4f}")
        
        return batch_results
    
    def _create_test_image_by_shape(self, shape: str) -> ImageData:
        """Create test image with specific shape."""
        image_data = np.zeros((32, 32), dtype=np.uint8)
        center = (16, 16)
        
        if shape == 'square':
            image_data[8:24, 8:24] = 200
        elif shape == 'circle':
            y, x = np.ogrid[:32, :32]
            mask = (x - center[0])**2 + (y - center[1])**2 <= 8**2
            image_data[mask] = 200
        elif shape == 'triangle':
            for i in range(8, 24):
                width = i - 8
                start = center[0] - width // 2
                end = center[0] + width // 2
                if start >= 0 and end < 32:
                    image_data[i, start:end] = 200
        elif shape == 'hexagon':
            # Simplified hexagon
            image_data[10:22, 6:26] = 200
            image_data[12:20, 4:28] = 200
        elif shape == 'cross':
            image_data[14:18, 6:26] = 200  # Horizontal bar
            image_data[6:26, 14:18] = 200  # Vertical bar
        
        # Add noise
        noise = np.random.normal(0, 5, (32, 32))
        image_data = np.clip(image_data.astype(float) + noise, 0, 255).astype(np.uint8)
        
        return ImageData.from_array(image_data, name=f"test_{shape}")
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        return {
            'pipeline_stats': self.stats.copy(),
            'component_stats': {
                'encoder_stats': self.encoder.get_statistics(),
                'designer_stats': self.designer.get_design_statistics(),
                'decoder_stats': self.decoder.get_decoder_statistics(),
                'simulator_stats': self.simulator.get_simulation_statistics(),
                'afm_processor_stats': self.afm_processor.get_processing_statistics()
            }
        }
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive pipeline performance report."""
        stats = self.get_pipeline_statistics()
        
        report = """
# DNA Origami Autoencoder Pipeline Report

## Overall Performance
"""
        
        pipeline_stats = stats['pipeline_stats']
        report += f"- Total pipeline runs: {pipeline_stats['total_runs']}\n"
        report += f"- Successful runs: {pipeline_stats['successful_runs']}\n"
        report += f"- Success rate: {pipeline_stats['successful_runs']/max(1, pipeline_stats['total_runs']):.1%}\n"
        report += f"- Average reconstruction accuracy: {pipeline_stats['average_reconstruction_accuracy']:.4f}\n"
        report += f"- Average pipeline time: {pipeline_stats['average_pipeline_time']:.2f}s\n\n"
        
        report += "## Component Performance\n\n"
        
        # Encoder stats
        encoder_stats = stats['component_stats']['encoder_stats']
        report += f"### DNA Encoder\n"
        report += f"- Images encoded: {encoder_stats['total_images_encoded']}\n"
        report += f"- Total bases generated: {encoder_stats['total_bases_generated']}\n"
        report += f"- Constraint violations: {encoder_stats['constraint_violations']}\n\n"
        
        # Decoder stats
        decoder_stats = stats['component_stats']['decoder_stats']
        report += f"### Transformer Decoder\n"
        report += f"- Structures decoded: {decoder_stats['structures_decoded']}\n"
        report += f"- Average decoding time: {decoder_stats.get('average_decoding_time', 0.0):.3f}s\n"
        report += f"- Model trained: {decoder_stats['is_trained']}\n\n"
        
        return report


def main():
    """Run example pipeline demonstration."""
    print("🧬 DNA Origami Autoencoder - Complete Pipeline Demo")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = CompletePipelineExample()
    
    # Run single pipeline example
    print("\n1. Single Pipeline Run:")
    result = pipeline.run_complete_pipeline(
        simulate_folding=True,
        generate_protocol=True,
        validate_reconstruction=True
    )
    
    if result['success']:
        print("✅ Single run completed successfully!")
        metrics = result.get('metrics', {})
        if metrics:
            print(f"   Reconstruction PSNR: {metrics['psnr']:.2f} dB")
    else:
        print(f"❌ Single run failed: {result.get('error', 'Unknown error')}")
    
    # Run batch experiment
    print("\n2. Batch Experiment:")
    batch_result = pipeline.run_batch_experiment(batch_size=3)
    
    # Generate report
    print("\n3. Performance Report:")
    report = pipeline.generate_comprehensive_report()
    print(report)
    
    print("\n🎉 Demo completed!")


if __name__ == "__main__":
    main()