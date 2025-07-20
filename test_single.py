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
    if len(sys.argv) > 1:
        # Extract file number from argument (e.g., "file03" or "test_data/file03.pdf")
        arg = sys.argv[1]
        if "file" in arg:
            if arg.startswith("test_data/"):
                input_file = arg
                file_num = arg.split("file")[1].split(".")[0]
            else:
                file_num = arg.replace("file", "")
                input_file = f"test_data/file{file_num.zfill(2)}.pdf"
        else:
            file_num = arg.zfill(2)
            input_file = f"test_data/file{file_num}.pdf"
        output_file = f"output/file{file_num.zfill(2)}_outline_test.json"
    else:
        input_file = "test_data/file01.pdf"
        output_file = "output/file01_outline_test.json"
    
    result = test_single_file(input_file, output_file)
