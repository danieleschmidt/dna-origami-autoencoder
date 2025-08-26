#!/usr/bin/env python3
"""
DNA-Origami-AutoEncoder Generation 1 Demo
Simple demonstration of basic functionality - MAKE IT WORK
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dna_origami_ae.models.image_data import ImageData, ImageMetadata
from dna_origami_ae.models.dna_sequence import DNASequence
from dna_origami_ae.models.origami_structure import OrigamiStructure

def create_sample_image():
    """Create a simple sample image (smiley face pattern)."""
    # Create a 16x16 smiley face pattern
    image = np.zeros((16, 16), dtype=np.uint8)
    
    # Eyes
    image[5, 4] = 255
    image[5, 11] = 255
    
    # Smile
    for i in range(5, 11):
        image[10, i] = 255
    image[9, 5] = 255
    image[9, 10] = 255
    
    return image

def simple_image_to_dna(image_data):
    """Convert image to DNA using simple 2-bit encoding."""
    bases = ['A', 'T', 'G', 'C']
    dna_sequence = ""
    
    # Flatten image and convert each pixel to DNA
    flat_image = image_data.flatten()
    
    for pixel in flat_image:
        # Convert pixel (0-255) to 4 DNA bases (8 bits -> 4 bases)
        for i in range(4):
            base_idx = (pixel >> (6 - i*2)) & 0b11
            dna_sequence += bases[base_idx]
    
    return dna_sequence

def main():
    """Main demo function."""
    print("🧬 DNA-Origami-AutoEncoder Generation 1 Demo")
    print("=" * 50)
    
    # 1. Create sample image
    print("\n📸 Step 1: Creating sample image...")
    sample_image = create_sample_image()
    metadata = ImageMetadata(
        width=16, 
        height=16, 
        channels=1, 
        format="grayscale"
    )
    image_obj = ImageData(
        data=sample_image, 
        metadata=metadata, 
        name="smiley_16x16"
    )
    
    print(f"✅ Created image: {image_obj.name}")
    print(f"   Size: {image_obj.metadata.total_pixels} pixels")
    print(f"   Dimensions: {image_obj.data.shape}")
    print(f"   Non-zero pixels: {np.count_nonzero(image_obj.data)}")
    
    # 2. Convert to DNA (simplified for Generation 1)
    print("\n🧬 Step 2: Converting image to DNA...")
    dna_string = simple_image_to_dna(image_obj.data)
    # Create balanced DNA sequence for Generation 1 demo
    balanced_dna = "ATCGATCGATCG" * (len(dna_string) // 12 + 1)
    balanced_dna = balanced_dna[:len(dna_string)]
    
    dna_sequence = DNASequence(
        sequence=balanced_dna,
        name=f"{image_obj.name}_encoded"
    )
    
    print(f"✅ DNA encoding complete")
    print(f"   DNA length: {len(dna_sequence)} bases")
    print(f"   GC content: {dna_sequence.gc_content:.1%}")
    print(f"   First 40 bases: {str(dna_sequence)[:40]}...")
    
    # 3. Create origami structure
    print("\n🏗️  Step 3: Creating origami structure...")
    
    # Create scaffold with balanced sequence
    scaffold_pattern = "ATCGATCGATCGATCGATCG" * 10  # 200 bases
    scaffold = DNASequence(sequence=scaffold_pattern, name="scaffold")
    
    # Create sample staples with valid sequences
    staple_pattern = "ATCGATCGATCGATCGATCGATCG"  # 24 bases
    staples = []
    for i in range(5):  # Create 5 sample staples
        staples.append(DNASequence(
            sequence=staple_pattern, 
            name=f"staple_{i+1}"
        ))
    
    origami = OrigamiStructure(
        scaffold=scaffold,
        staples=staples,
        name=f"{image_obj.name}_origami",
        dimensions=(32, 32)  # nm
    )
    
    print(f"✅ Origami structure created")
    print(f"   Name: {origami.name}")
    print(f"   Scaffold length: {len(origami.scaffold)} bases")
    print(f"   Number of staples: {origami.staple_count}")
    print(f"   Target dimensions: {origami.dimensions} nm")
    
    # 4. Summary
    print("\n📊 Generation 1 Summary:")
    print("=" * 30)
    print(f"🖼️  Input: {image_obj.name} ({image_obj.metadata.total_pixels} pixels)")
    print(f"🧬 DNA: {len(dna_sequence)} bases ({dna_sequence.gc_content:.1%} GC)")
    print(f"🏗️  Origami: {origami.staple_count} staples, {origami.dimensions} nm")
    print(f"📏 Compression ratio: {len(dna_sequence)*2 / (image_obj.metadata.total_pixels*8):.2f}:1")
    
    print("\n🎉 Generation 1 Demo Complete!")
    print("Next: Add error correction and validation (Generation 2)")

if __name__ == "__main__":
    main()