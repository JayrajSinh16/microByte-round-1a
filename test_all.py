#!/usr/bin/env python3
"""
Test all PDF files in the test_data directory
"""
import os
import sys
import json
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / 'src'
sys.path.insert(0, str(src_dir))

from main import PDFOutlineExtractor

def test_all_pdfs():
    extractor = PDFOutlineExtractor()
    test_data_dir = current_dir / 'test_data'
    output_dir = current_dir / 'output'
    output_dir.mkdir(exist_ok=True)
    
    pdf_files = list(test_data_dir.glob('*.pdf'))
    
    print(f"🔄 Testing {len(pdf_files)} PDF files:\n")
    
    results_summary = []
    
    for pdf_file in pdf_files:
        print(f"📄 Processing: {pdf_file.name}")
        
        try:
            result = extractor.process_pdf(str(pdf_file))
            
            # Save result
            output_file = output_dir / f"{pdf_file.stem}_outline.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            # Summary
            title = result.get('title', 'N/A')
            outline_count = len(result.get('outline', []))
            
            print(f"  ✅ Title: {title[:50]}{'...' if len(title) > 50 else ''}")
            print(f"  📋 Headings: {outline_count}")
            
            results_summary.append({
                'file': pdf_file.name,
                'title': title,
                'headings': outline_count,
                'success': True
            })
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            results_summary.append({
                'file': pdf_file.name,
                'error': str(e),
                'success': False
            })
        
        print()  # Empty line for readability
    
    # Overall summary
    successful = sum(1 for r in results_summary if r['success'])
    total = len(results_summary)
    
    print(f"\n{'='*50}")
    print(f"📊 SUMMARY: {successful}/{total} files processed successfully")
    print(f"{'='*50}")
    
    for result in results_summary:
        if result['success']:
            print(f"✅ {result['file']:15} - {result['headings']:2d} headings")
        else:
            print(f"❌ {result['file']:15} - ERROR")

if __name__ == "__main__":
    test_all_pdfs()
