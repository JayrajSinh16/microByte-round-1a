#!/usr/bin/env python3
"""
Run PDF outline extraction on input PDFs
"""
import sys
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / 'src'
sys.path.insert(0, str(src_dir))

from main import PDFOutlineExtractor

def main():
    print("🚀 Starting PDF Outline Extraction...")
    
    # Create extractor
    extractor = PDFOutlineExtractor()
    
    # Define input and output directories
    input_dir = current_dir / 'input'
    output_dir = current_dir / 'output'
    
    print(f"📁 Input directory: {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    
    # Process all PDFs
    extractor.process_directory(str(input_dir), str(output_dir))
    
    print("✅ Processing completed!")
    print(f"📄 Check the 'output' directory for results")

if __name__ == "__main__":
    main()
