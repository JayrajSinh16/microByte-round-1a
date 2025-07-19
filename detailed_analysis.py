#!/usr/bin/env python3
"""
Detailed breakdown of accuracy issues
"""
import json
from pathlib import Path

def analyze_differences():
    expected_dir = Path("output/expected_output")
    actual_dir = Path("output")
    
    print("🔍 DETAILED ACCURACY BREAKDOWN")
    print("="*70)
    
    for i in range(1, 6):
        file_name = f"file{i:02d}"
        expected_file = expected_dir / f"{file_name}.json"
        actual_file = actual_dir / f"{file_name}_outline.json"
        
        if not expected_file.exists() or not actual_file.exists():
            continue
            
        with open(expected_file, 'r', encoding='utf-8') as f:
            expected = json.load(f)
        with open(actual_file, 'r', encoding='utf-8') as f:
            actual = json.load(f)
            
        print(f"\n📄 {file_name}.pdf ANALYSIS:")
        print("-" * 50)
        
        # Title comparison
        exp_title = expected.get('title', '').strip()
        act_title = actual.get('title', '').strip()
        
        print(f"📝 TITLE COMPARISON:")
        print(f"   Expected: '{exp_title}'")
        print(f"   Actual:   '{act_title}'")
        print(f"   Match:    {'✅ YES' if exp_title.lower().replace(' ', '') == act_title.lower().replace(' ', '') else '❌ NO'}")
        
        # Outline comparison
        exp_outline = expected.get('outline', [])
        act_outline = actual.get('outline', [])
        
        print(f"📋 OUTLINE COMPARISON:")
        print(f"   Expected headings: {len(exp_outline)}")
        print(f"   Actual headings:   {len(act_outline)}")
        
        if exp_outline:
            print(f"   Expected outline:")
            for j, heading in enumerate(exp_outline[:3]):
                print(f"     {j+1}. {heading.get('level', 'N/A')}: '{heading.get('text', 'N/A')}' (page {heading.get('page', 'N/A')})")
            if len(exp_outline) > 3:
                print(f"     ... and {len(exp_outline) - 3} more")
        else:
            print(f"   Expected: EMPTY OUTLINE")
            
        if act_outline:
            print(f"   Actual outline:")
            for j, heading in enumerate(act_outline[:3]):
                print(f"     {j+1}. {heading.get('level', 'N/A')}: '{heading.get('text', 'N/A')}' (page {heading.get('page', 'N/A')})")
            if len(act_outline) > 3:
                print(f"     ... and {len(act_outline) - 3} more")
        else:
            print(f"   Actual: EMPTY OUTLINE")

    print("\n" + "="*70)
    print("🎯 ACCURACY SUMMARY:")
    print("="*70)
    print("Current accuracy: 45.5%")
    print("Target accuracy:  80%")
    print("Gap to close:     34.5%")
    
    print("\n📊 MAIN ISSUES IDENTIFIED:")
    print("1. ❌ file01.pdf: Expected empty outline, but system found 17 headings")
    print("2. ❌ file05.pdf: Expected empty title, but system extracted 'RSVP: ----------------'")
    print("3. ⚠️  Title extraction: Some titles are partial or incorrect")
    print("4. ⚠️  Outline precision: System finds more headings than expected")
    
    print("\n🔧 POTENTIAL IMPROVEMENTS:")
    print("1. Better title normalization (remove extra spaces)")
    print("2. More conservative heading detection (fewer false positives)")
    print("3. Handle cases where expected outline is empty")
    print("4. Improve handling of non-standard document structures")

if __name__ == "__main__":
    analyze_differences()
