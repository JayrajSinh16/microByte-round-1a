#!/usr/bin/env python3
"""
Simple test script to demonstrate the PDF Outline Extractor
"""
import os
import sys
import json
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / 'src'
sys.path.insert(0, str(src_dir))

try:
    from main import PDFOutlineExtractor
    print("✅ Successfully imported PDFOutlineExtractor")
    
    # Create extractor instance
    extractor = PDFOutlineExtractor()
    print("✅ Successfully created extractor instance")
    
    # Check for test PDFs
    test_data_dir = current_dir / 'test_data'
    pdf_files = list(test_data_dir.glob('*.pdf'))
    
    if pdf_files:
        print(f"📁 Found {len(pdf_files)} test PDF files:")
        for pdf_file in pdf_files:
            print(f"  - {pdf_file.name}")
        
        # Process the first PDF file
        first_pdf = pdf_files[0]
        print(f"\n🔄 Processing: {first_pdf.name}")
        
        try:
            result = extractor.process_pdf(str(first_pdf))
            print("✅ Successfully processed PDF!")
            print(f"📄 Title: {result.get('title', 'N/A')}")
            print(f"📋 Found {len(result.get('outline', []))} headings")
            
            # Show first few headings
            outline = result.get('outline', [])
            if outline:
                print("\n📋 First few headings:")
                for i, heading in enumerate(outline[:5]):
                    print(f"  {heading['level']}: {heading['text']} (page {heading['page']})")
                if len(outline) > 5:
                    print(f"  ... and {len(outline) - 5} more")
            
            # Save result to output
            output_dir = current_dir / 'output'
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / f"{first_pdf.stem}_outline.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Saved result to: {output_file}")
            
        except Exception as e:
            print(f"❌ Error processing PDF: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        print("⚠️  No test PDF files found in test_data directory")
        print("   Add some PDF files to test_data/ to see the extractor in action")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Check that all required modules are available in the src directory")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
