#!/usr/bin/env python3
"""
Simple test script for individual PDF files
"""
import sys
import json
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, 'src')

from main import PDFOutlineExtractor

def test_single_file(input_file, output_file):
    """Test a single PDF file"""
    extractor = PDFOutlineExtractor()
    
    try:
        result = extractor.process_pdf(input_file)
        
        # Save result
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully processed {input_file}")
        print(f"Output saved to {output_file}")
        
        # Print summary
        print(f"Title: {result.get('title', 'N/A')}")
        print(f"Headings found: {len(result.get('outline', []))}")
        
        return result
        
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return None

if __name__ == "__main__":
    input_file = "test_data/file01.pdf"
    output_file = "output/file01_outline_test.json"
    
    result = test_single_file(input_file, output_file)
