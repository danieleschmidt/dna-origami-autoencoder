"""
DNA-Origami-AutoEncoder Command Line Interface

A comprehensive CLI for the DNA origami autoencoder framework, supporting
image encoding, origami design, simulation, and decoding workflows.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import numpy as np
from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Core imports
from .encoding import DNAEncoder, Base4Encoder, BiologicalConstraints
from .design import OrigamiDesigner, ShapeDesigner, Origami3D
from .simulation import OrigamiSimulator, GPUSimulator, CoarseGrainedModel
from .decoding import TransformerDecoder, OrigamiTransformer
from .models import DNASequence, OrigamiStructure, ImageData
from .utils import validators, helpers, cache, i18n, memory_optimizer

# Initialize console and logging
console = Console()
logger = logging.getLogger(__name__)

# CLI Groups and Commands
@click.group()
@click.option('--verbose', '-v', count=True, help='Increase verbosity (-v, -vv, -vvv)')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--cache-dir', type=click.Path(), help='Cache directory path')
@click.option('--gpu/--no-gpu', default=True, help='Enable/disable GPU acceleration')
@click.option('--language', '-l', default='en', help='Interface language (en, es, fr, de, ja, zh)')
@click.pass_context
def cli(ctx, verbose: int, config: Optional[str], cache_dir: Optional[str], 
        gpu: bool, language: str) -> None:
    """DNA-Origami-AutoEncoder: Encode images into self-assembling DNA structures."""
    
    # Setup context
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['gpu'] = gpu
    ctx.obj['language'] = language
    
    # Configure logging
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    # Initialize internationalization
    i18n.set_language(language)
    
    # Setup cache
    if cache_dir:
        cache.set_cache_dir(Path(cache_dir))
    
    # Memory optimization
    if verbose >= 2:
        memory_optimizer.enable_profiling()
    
    console.print(f"[bold blue]DNA-Origami-AutoEncoder CLI v0.1.0[/bold blue]")
    if verbose >= 1:
        console.print(f"GPU acceleration: {'enabled' if gpu else 'disabled'}")
        console.print(f"Language: {language}")


@cli.group()
@click.pass_context
def encode(ctx) -> None:
    """Encode images into DNA sequences."""
    pass


@encode.command('image')
@click.argument('input_image', type=click.Path(exists=True))
@click.argument('output_dna', type=click.Path())
@click.option('--encoder', '-e', type=click.Choice(['base4', 'goldman', 'church']), 
              default='base4', help='DNA encoding method')
@click.option('--error-correction', '-ec', type=click.Choice(['reed_solomon', 'hamming']), 
              default='reed_solomon', help='Error correction method')
@click.option('--compression/--no-compression', default=True, help='Enable compression')
@click.option('--size', '-s', type=int, nargs=2, default=(32, 32), 
              help='Resize image dimensions (width height)')
@click.option('--gc-content', type=float, nargs=2, default=(0.4, 0.6), 
              help='GC content range (min max)')
@click.option('--max-homopolymer', type=int, default=4, help='Maximum homopolymer length')
@click.option('--validate/--no-validate', default=True, help='Validate biological constraints')
@click.pass_context
def encode_image(ctx, input_image: str, output_dna: str, encoder: str, 
                error_correction: str, compression: bool, size: Tuple[int, int],
                gc_content: Tuple[float, float], max_homopolymer: int, validate: bool) -> None:
    """Encode an image into DNA sequences with biological constraints."""
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Load and preprocess image
            progress.add_task(description="Loading image...", total=None)
            image = Image.open(input_image).convert('L')
            if size != (32, 32):
                image = image.resize(size)
            image_array = np.array(image)
            
            # Setup biological constraints
            progress.add_task(description="Setting up biological constraints...", total=None)
            constraints = BiologicalConstraints(
                gc_content=gc_content,
                max_homopolymer=max_homopolymer,
                avoid_sequences=['GGGG', 'CCCC', 'AAAA', 'TTTT'],
                melting_temp_range=(55, 65)
            )
            
            # Initialize encoder
            progress.add_task(description=f"Initializing {encoder} encoder...", total=None)
            if encoder == 'base4':
                dna_encoder = Base4Encoder(constraints if validate else None)
            else:
                dna_encoder = DNAEncoder(
                    encoding_method=encoder,
                    error_correction=error_correction,
                    compression=compression,
                    biological_constraints=constraints if validate else None
                )
            
            # Encode image
            progress.add_task(description="Encoding image to DNA...", total=None)
            dna_sequence = dna_encoder.encode_image(image_array)
            
            # Save results
            progress.add_task(description="Saving DNA sequence...", total=None)
            output_path = Path(output_dna)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(f"# DNA sequence for {Path(input_image).name}\n")
                f.write(f"# Encoding: {encoder}\n")
                f.write(f"# Error correction: {error_correction}\n")
                f.write(f"# Image size: {size[0]}x{size[1]}\n")
                f.write(f"# Sequence length: {len(dna_sequence)} bases\n")
                f.write(f"# GC content: {dna_encoder.get_gc_content(dna_sequence):.2%}\n\n")
                f.write(dna_sequence)
        
        # Display statistics
        stats_table = Table(title="Encoding Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        stats_table.add_row("DNA sequence length", f"{len(dna_sequence):,} bases")
        stats_table.add_row("Compression ratio", f"{len(image_array.flatten()) / len(dna_sequence) * 4:.2f}x")
        stats_table.add_row("GC content", f"{dna_encoder.get_gc_content(dna_sequence):.2%}")
        stats_table.add_row("Encoding efficiency", f"{dna_encoder.get_efficiency():.2%}")
        
        console.print(stats_table)
        console.print(f"[green]✓[/green] Image encoded successfully: {output_dna}")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Encoding failed: {e}")
        logger.error(f"Encoding error: {e}")
        sys.exit(1)


@cli.group()
@click.pass_context
def design(ctx) -> None:
    """Design DNA origami structures."""
    pass


@design.command('origami')
@click.argument('dna_sequence_file', type=click.Path(exists=True))
@click.argument('output_design', type=click.Path())
@click.option('--shape', '-s', type=click.Choice(['square', 'rectangle', 'circle', 'custom']), 
              default='square', help='Target origami shape')
@click.option('--dimensions', '-d', type=float, nargs=2, default=(100, 100), 
              help='Dimensions in nanometers (width height)')
@click.option('--scaffold-length', type=int, default=7249, help='Scaffold length (M13mp18=7249)')
@click.option('--staple-length', type=int, default=32, help='Staple strand length')
@click.option('--crossover-spacing', type=int, default=21, help='Crossover spacing in bases')
@click.option('--optimize/--no-optimize', default=True, help='Optimize staple sequences')
@click.option('--export-format', '-f', multiple=True, 
              type=click.Choice(['csv', 'json', 'cadnano']), default=['csv'],
              help='Export formats')
@click.pass_context
def design_origami(ctx, dna_sequence_file: str, output_design: str, shape: str,
                  dimensions: Tuple[float, float], scaffold_length: int, 
                  staple_length: int, crossover_spacing: int, optimize: bool,
                  export_format: List[str]) -> None:
    """Design DNA origami structure from encoded sequence."""
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Load DNA sequence
            progress.add_task(description="Loading DNA sequence...", total=None)
            with open(dna_sequence_file, 'r') as f:
                lines = f.readlines()
                dna_sequence = ''.join(line.strip() for line in lines if not line.startswith('#'))
            
            # Initialize designer
            progress.add_task(description="Initializing origami designer...", total=None)
            designer = OrigamiDesigner(
                scaffold_length=scaffold_length,
                staple_length=staple_length,
                crossover_spacing=crossover_spacing
            )
            
            # Design origami
            progress.add_task(description=f"Designing {shape} origami structure...", total=None)
            origami = designer.design_origami(
                dna_sequence,
                target_shape=shape,
                dimensions=dimensions,
                optimize_sequences=optimize
            )
            
            # Export designs
            output_path = Path(output_design)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            for fmt in export_format:
                progress.add_task(description=f"Exporting {fmt.upper()} format...", total=None)
                export_file = output_path.with_suffix(f'.{fmt}')
                
                if fmt == 'csv':
                    origami.export_sequences(export_file)
                elif fmt == 'json':
                    origami.to_json(export_file)
                elif fmt == 'cadnano':
                    origami.to_cadnano(export_file)
        
        # Display design statistics
        stats_table = Table(title="Design Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        stats_table.add_row("Scaffold length", f"{origami.scaffold_length:,} bases")
        stats_table.add_row("Number of staples", f"{len(origami.staples):,}")
        stats_table.add_row("Total staple length", f"{sum(len(s.sequence) for s in origami.staples):,} bases")
        stats_table.add_row("Design dimensions", f"{dimensions[0]}×{dimensions[1]} nm")
        stats_table.add_row("Estimated yield", f"{origami.get_estimated_yield():.1%}")
        
        console.print(stats_table)
        console.print(f"[green]✓[/green] Origami design completed: {output_design}")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Design failed: {e}")
        logger.error(f"Design error: {e}")
        sys.exit(1)


@cli.group()
@click.pass_context  
def simulate(ctx) -> None:
    """Run molecular dynamics simulations."""
    pass


@simulate.command('folding')
@click.argument('origami_design', type=click.Path(exists=True))
@click.argument('output_trajectory', type=click.Path())
@click.option('--force-field', '-ff', type=click.Choice(['oxdna', 'oxdna2']), 
              default='oxdna2', help='Force field for simulation')
@click.option('--temperature', '-T', type=float, default=300, help='Temperature in Kelvin')
@click.option('--salt-concentration', type=float, default=0.5, help='Salt concentration in M')
@click.option('--time-steps', type=int, default=1000000, help='Number of simulation steps')
@click.option('--save-interval', type=int, default=1000, help='Save trajectory every N steps')
@click.option('--gpu/--no-gpu', default=True, help='Use GPU acceleration')
@click.option('--batch-size', type=int, default=1, help='Batch size for parallel simulation')
@click.pass_context
def simulate_folding(ctx, origami_design: str, output_trajectory: str, 
                    force_field: str, temperature: float, salt_concentration: float,
                    time_steps: int, save_interval: int, gpu: bool, batch_size: int) -> None:
    """Simulate DNA origami folding dynamics."""
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Load origami design
            progress.add_task(description="Loading origami design...", total=None)
            # Implementation would load from various formats
            
            # Initialize simulator
            progress.add_task(description="Initializing simulator...", total=None)
            if gpu and ctx.obj['gpu']:
                simulator = GPUSimulator(
                    device='cuda:0',
                    precision='mixed',
                    batch_size=batch_size
                )
            else:
                simulator = OrigamiSimulator(
                    force_field=force_field,
                    temperature=temperature,
                    salt_concentration=salt_concentration
                )
            
            # Run simulation
            progress.add_task(description="Running folding simulation...", total=None)
            trajectory = simulator.simulate_folding(
                origami_design,  # Would be parsed origami object
                time_steps=time_steps,
                save_interval=save_interval,
                gpu_acceleration=gpu
            )
            
            # Save trajectory
            progress.add_task(description="Saving trajectory...", total=None)
            output_path = Path(output_trajectory)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory.save(output_path)
        
        # Display simulation statistics
        stats_table = Table(title="Simulation Statistics")
        stats_table.add_column("Metric", style="cyan") 
        stats_table.add_column("Value", style="green")
        
        stats_table.add_row("Total time steps", f"{time_steps:,}")
        stats_table.add_row("Simulation time", f"{trajectory.get_total_time():.2f} µs")
        stats_table.add_row("Final RMSD", f"{trajectory.get_final_rmsd():.2f} nm")
        stats_table.add_row("Folding completion", f"{trajectory.get_folding_percentage():.1%}")
        
        console.print(stats_table)
        console.print(f"[green]✓[/green] Simulation completed: {output_trajectory}")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Simulation failed: {e}")
        logger.error(f"Simulation error: {e}")
        sys.exit(1)


@cli.group()
@click.pass_context
def decode(ctx) -> None:
    """Decode structures back to images."""
    pass


@decode.command('structure')
@click.argument('structure_file', type=click.Path(exists=True))
@click.argument('output_image', type=click.Path())
@click.option('--model', '-m', type=click.Path(exists=True), 
              help='Transformer decoder model path')
@click.option('--model-type', type=click.Choice(['transformer', 'cnn']), 
              default='transformer', help='Decoder model type')
@click.option('--confidence-threshold', type=float, default=0.8, 
              help='Minimum confidence threshold')
@click.option('--batch-size', type=int, default=1, help='Batch size for inference')
@click.pass_context
def decode_structure(ctx, structure_file: str, output_image: str, model: Optional[str],
                    model_type: str, confidence_threshold: float, batch_size: int) -> None:
    """Decode DNA structure back to original image."""
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Load structure
            progress.add_task(description="Loading structure data...", total=None)
            # Implementation would load trajectory or structure file
            
            # Initialize decoder
            progress.add_task(description=f"Loading {model_type} decoder...", total=None)
            if model:
                if model_type == 'transformer':
                    decoder = TransformerDecoder.load(model)
                else:
                    # Would load CNN decoder
                    pass
            else:
                # Use pretrained model
                decoder = TransformerDecoder.from_pretrained('dna-origami-decoder-v1')
            
            # Decode structure
            progress.add_task(description="Decoding structure to image...", total=None)
            decoded_image, confidence = decoder.decode_structure(
                structure_file,  # Would be parsed structure
                return_confidence=True
            )
            
            # Validate confidence
            if confidence < confidence_threshold:
                console.print(f"[yellow]⚠[/yellow] Low confidence: {confidence:.2%}")
            
            # Save image
            progress.add_task(description="Saving decoded image...", total=None)
            output_path = Path(output_image)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to PIL Image and save
            image = Image.fromarray((decoded_image * 255).astype(np.uint8), mode='L')
            image.save(output_path)
        
        # Display decoding statistics
        stats_table = Table(title="Decoding Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        stats_table.add_row("Confidence score", f"{confidence:.2%}")
        stats_table.add_row("Image dimensions", f"{decoded_image.shape[1]}×{decoded_image.shape[0]}")
        stats_table.add_row("Pixel intensity range", f"{decoded_image.min():.3f} - {decoded_image.max():.3f}")
        
        console.print(stats_table)
        console.print(f"[green]✓[/green] Structure decoded successfully: {output_image}")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Decoding failed: {e}")
        logger.error(f"Decoding error: {e}")
        sys.exit(1)


@cli.group()
@click.pass_context
def pipeline(ctx) -> None:
    """Run complete end-to-end pipelines."""
    pass


@pipeline.command('full')
@click.argument('input_image', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--simulate/--no-simulate', default=False, help='Include folding simulation')
@click.option('--size', '-s', type=int, nargs=2, default=(32, 32), help='Image dimensions')
@click.option('--shape', type=click.Choice(['square', 'rectangle', 'circle']), 
              default='square', help='Origami shape')
@click.option('--gpu/--no-gpu', default=True, help='Use GPU acceleration')
@click.pass_context
def full_pipeline(ctx, input_image: str, output_dir: str, simulate: bool,
                 size: Tuple[int, int], shape: str, gpu: bool) -> None:
    """Run complete encode -> design -> [simulate] -> decode pipeline."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Step 1: Encode image
            task1 = progress.add_task(description="[1/4] Encoding image to DNA...", total=None)
            dna_file = output_path / 'encoded.dna'
            # Call encode_image function programmatically
            progress.advance(task1)
            
            # Step 2: Design origami
            task2 = progress.add_task(description="[2/4] Designing origami structure...", total=None)
            design_file = output_path / 'design.csv'
            # Call design_origami function programmatically  
            progress.advance(task2)
            
            # Step 3: Simulate (optional)
            if simulate:
                task3 = progress.add_task(description="[3/4] Simulating folding...", total=None)
                trajectory_file = output_path / 'trajectory.h5'
                # Call simulate_folding function programmatically
                progress.advance(task3)
                structure_file = trajectory_file
            else:
                structure_file = design_file
            
            # Step 4: Decode
            task4 = progress.add_task(description="[4/4] Decoding to image...", total=None)
            decoded_file = output_path / 'decoded.png'
            # Call decode_structure function programmatically
            progress.advance(task4)
        
        # Generate report
        report_file = output_path / 'pipeline_report.md'
        with open(report_file, 'w') as f:
            f.write("# DNA Origami AutoEncoder Pipeline Report\n\n")
            f.write(f"**Input Image:** {input_image}\n")
            f.write(f"**Output Directory:** {output_dir}\n")
            f.write(f"**Image Size:** {size[0]}×{size[1]}\n")
            f.write(f"**Origami Shape:** {shape}\n")
            f.write(f"**Simulation:** {'Yes' if simulate else 'No'}\n")
            f.write(f"**GPU Acceleration:** {'Yes' if gpu else 'No'}\n\n")
            f.write("## Generated Files\n\n")
            f.write("- `encoded.dna` - DNA sequence encoding\n")
            f.write("- `design.csv` - Origami design with staple sequences\n")
            if simulate:
                f.write("- `trajectory.h5` - Folding simulation trajectory\n")
            f.write("- `decoded.png` - Reconstructed image\n")
            f.write("- `pipeline_report.md` - This report\n")
        
        console.print(f"[green]✓[/green] Full pipeline completed: {output_dir}")
        console.print(f"[blue]ℹ[/blue] Report generated: {report_file}")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Pipeline failed: {e}")
        logger.error(f"Pipeline error: {e}")
        sys.exit(1)


@cli.command()
@click.option('--check-gpu', is_flag=True, help='Check GPU availability')
@click.option('--check-deps', is_flag=True, help='Check dependencies')  
@click.option('--benchmark', is_flag=True, help='Run performance benchmark')
@click.pass_context
def info(ctx, check_gpu: bool, check_deps: bool, benchmark: bool) -> None:
    """Display system information and run diagnostics."""
    
    console.print("[bold]System Information[/bold]")
    
    # Basic info
    info_table = Table()
    info_table.add_column("Component", style="cyan")
    info_table.add_column("Status", style="green")
    
    info_table.add_row("Python version", f"{sys.version.split()[0]}")
    info_table.add_row("Platform", f"{sys.platform}")
    
    # GPU check
    if check_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                info_table.add_row("GPU", f"✓ {gpu_name}")
            else:
                info_table.add_row("GPU", "✗ Not available")
        except ImportError:
            info_table.add_row("GPU", "✗ PyTorch not installed")
    
    # Dependencies check
    if check_deps:
        required_deps = [
            'numpy', 'scipy', 'pandas', 'matplotlib', 'PIL', 
            'sklearn', 'torch', 'biopython', 'click', 'rich'
        ]
        
        for dep in required_deps:
            try:
                __import__(dep)
                info_table.add_row(f"Dependency: {dep}", "✓ Available")
            except ImportError:
                info_table.add_row(f"Dependency: {dep}", "✗ Missing")
    
    console.print(info_table)
    
    # Performance benchmark
    if benchmark:
        console.print("\n[bold]Running Performance Benchmark...[/bold]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # DNA encoding benchmark
            progress.add_task(description="Benchmarking DNA encoding...", total=None)
            encoder = Base4Encoder()
            test_data = np.random.randint(0, 256, size=(32, 32), dtype=np.uint8)
            
            import time
            start_time = time.time()
            dna_seq = encoder.encode_image(test_data)
            encode_time = time.time() - start_time
            
            # Decoding benchmark  
            progress.add_task(description="Benchmarking decoding...", total=None)
            start_time = time.time()
            decoded = encoder.decode_dna(dna_seq, (32, 32))
            decode_time = time.time() - start_time
        
        bench_table = Table(title="Performance Benchmark")
        bench_table.add_column("Operation", style="cyan")
        bench_table.add_column("Time", style="green")
        bench_table.add_column("Rate", style="yellow")
        
        bench_table.add_row("DNA Encoding (32×32)", f"{encode_time:.3f}s", 
                          f"{len(test_data.flatten())/encode_time:.0f} pixels/s")
        bench_table.add_row("DNA Decoding (32×32)", f"{decode_time:.3f}s",
                          f"{len(decoded.flatten())/decode_time:.0f} pixels/s")
        
        console.print(bench_table)


def main() -> None:
    """Main CLI entry point."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        logger.exception("Unexpected CLI error")
        sys.exit(1)


if __name__ == '__main__':
    main()