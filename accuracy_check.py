#!/usr/bin/env python3
"""
Accuracy comparison between expected output and actual output
"""
import json
from pathlib import Path
import difflib
from typing import Dict, List, Tuple

def normalize_text(text: str) -> str:
    """Normalize text for comparison by removing extra spaces and converting to lowercase"""
    return ' '.join(text.strip().split()).lower()

def calculate_title_accuracy(expected_title: str, actual_title: str) -> float:
    """Calculate title accuracy using string similarity"""
    expected_norm = normalize_text(expected_title)
    actual_norm = normalize_text(actual_title)
    
    # Use sequence matcher for similarity
    similarity = difflib.SequenceMatcher(None, expected_norm, actual_norm).ratio()
    return similarity

def calculate_outline_accuracy(expected_outline: List[Dict], actual_outline: List[Dict]) -> Dict:
    """Calculate outline accuracy by comparing headings"""
    
    if not expected_outline and not actual_outline:
        return {"accuracy": 1.0, "total_expected": 0, "total_actual": 0, "matches": 0, "recall": 1.0, "precision": 1.0, "details": "Both outlines empty - perfect match"}
    
    if not expected_outline:
        return {"accuracy": 0.0, "total_expected": 0, "total_actual": len(actual_outline), "matches": 0, "recall": 0.0, "precision": 0.0, "details": f"Expected empty, got {len(actual_outline)} headings"}
    
    if not actual_outline:
        return {"accuracy": 0.0, "total_expected": len(expected_outline), "total_actual": 0, "matches": 0, "recall": 0.0, "precision": 0.0, "details": f"Expected {len(expected_outline)} headings, got empty"}
    
    # Compare each heading
    matches = 0
    total_checks = 0
    details = []
    
    # Create a mapping of expected headings by page and level
    expected_by_page = {}
    for exp_heading in expected_outline:
        page = exp_heading.get('page', 0)
        if page not in expected_by_page:
            expected_by_page[page] = []
        expected_by_page[page].append(exp_heading)
    
    # Check how many actual headings match expected ones
    for actual_heading in actual_outline:
        actual_page = actual_heading.get('page', 0)
        actual_text = normalize_text(actual_heading.get('text', ''))
        actual_level = actual_heading.get('level', '')
        
        best_match_score = 0
        best_match = None
        
        # Look for matches on the same page first, then nearby pages
        for search_page in [actual_page, actual_page - 1, actual_page + 1]:
            if search_page in expected_by_page:
                for exp_heading in expected_by_page[search_page]:
                    exp_text = normalize_text(exp_heading.get('text', ''))
                    exp_level = exp_heading.get('level', '')
                    
                    # Calculate match score
                    text_similarity = difflib.SequenceMatcher(None, actual_text, exp_text).ratio()
                    level_match = 1.0 if actual_level == exp_level else 0.5
                    page_penalty = 0.9 if search_page != actual_page else 1.0
                    
                    score = (text_similarity * 0.7 + level_match * 0.3) * page_penalty
                    
                    if score > best_match_score:
                        best_match_score = score
                        best_match = exp_heading
        
        total_checks += 1
        if best_match_score > 0.6:  # Threshold for considering it a match
            matches += 1
            details.append(f"✓ Match: '{actual_heading.get('text', '')[:30]}...' (score: {best_match_score:.2f})")
        else:
            details.append(f"✗ No match: '{actual_heading.get('text', '')[:30]}...'")
    
    # Also check for missing expected headings
    found_expected = 0
    for exp_heading in expected_outline:
        exp_text = normalize_text(exp_heading.get('text', ''))
        exp_page = exp_heading.get('page', 0)
        
        for actual_heading in actual_outline:
            actual_text = normalize_text(actual_heading.get('text', ''))
            actual_page = actual_heading.get('page', 0)
            
            text_similarity = difflib.SequenceMatcher(None, exp_text, actual_text).ratio()
            page_close = abs(actual_page - exp_page) <= 1
            
            if text_similarity > 0.6 and page_close:
                found_expected += 1
                break
    
    # Calculate overall accuracy
    recall = found_expected / len(expected_outline) if expected_outline else 0
    precision = matches / len(actual_outline) if actual_outline else 0
    
    # F1 score as overall accuracy
    if recall + precision > 0:
        accuracy = 2 * (recall * precision) / (recall + precision)
    else:
        accuracy = 0
    
    return {
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "matches": matches,
        "total_actual": len(actual_outline),
        "total_expected": len(expected_outline),
        "details": details[:5]  # First 5 details
    }

def compare_files():
    """Compare all files and calculate accuracy"""
    
    expected_dir = Path("output/expected_output")
    actual_dir = Path("output")
    
    if not expected_dir.exists():
        print("❌ Expected output directory not found!")
        return
    
    print("🔍 ACCURACY COMPARISON REPORT")
    print("="*60)
    
    file_accuracies = []
    overall_title_accuracy = 0
    overall_outline_accuracy = 0
    
    # Get all expected files
    expected_files = list(expected_dir.glob("file*.json"))
    
    for expected_file in expected_files:
        file_base = expected_file.stem  # e.g., "file01"
        actual_file = actual_dir / f"{file_base}_outline.json"
        
        print(f"\n📄 Comparing {file_base}.pdf:")
        print("-" * 40)
        
        if not actual_file.exists():
            print(f"❌ Actual output file not found: {actual_file}")
            continue
        
        try:
            # Load files
            with open(expected_file, 'r', encoding='utf-8') as f:
                expected = json.load(f)
            
            with open(actual_file, 'r', encoding='utf-8') as f:
                actual = json.load(f)
            
            # Compare titles
            title_accuracy = calculate_title_accuracy(
                expected.get('title', ''), 
                actual.get('title', '')
            )
            
            # Compare outlines
            outline_result = calculate_outline_accuracy(
                expected.get('outline', []),
                actual.get('outline', [])
            )
            
            print(f"📝 Title Accuracy: {title_accuracy:.1%}")
            print(f"   Expected: '{expected.get('title', '')[:50]}...'")
            print(f"   Actual:   '{actual.get('title', '')[:50]}...'")
            
            print(f"📋 Outline Accuracy: {outline_result['accuracy']:.1%}")
            print(f"   Expected headings: {outline_result['total_expected']}")
            print(f"   Actual headings: {outline_result['total_actual']}")
            print(f"   Matches found: {outline_result.get('matches', 0)}")
            print(f"   Recall: {outline_result.get('recall', 0):.1%}")
            print(f"   Precision: {outline_result.get('precision', 0):.1%}")
            
            # Overall file accuracy (weighted average)
            file_accuracy = (title_accuracy * 0.3 + outline_result['accuracy'] * 0.7)
            file_accuracies.append({
                'file': file_base,
                'title_accuracy': title_accuracy,
                'outline_accuracy': outline_result['accuracy'],
                'overall_accuracy': file_accuracy
            })
            
            overall_title_accuracy += title_accuracy
            overall_outline_accuracy += outline_result['accuracy']
            
            print(f"🎯 Overall File Accuracy: {file_accuracy:.1%}")
            
        except Exception as e:
            print(f"❌ Error comparing {file_base}: {e}")
    
    # Calculate overall statistics
    if file_accuracies:
        num_files = len(file_accuracies)
        avg_title_accuracy = overall_title_accuracy / num_files
        avg_outline_accuracy = overall_outline_accuracy / num_files
        avg_overall_accuracy = sum(f['overall_accuracy'] for f in file_accuracies) / num_files
        
        print("\n" + "="*60)
        print("📊 OVERALL ACCURACY SUMMARY")
        print("="*60)
        print(f"Average Title Accuracy:   {avg_title_accuracy:.1%}")
        print(f"Average Outline Accuracy: {avg_outline_accuracy:.1%}")
        print(f"Average Overall Accuracy: {avg_overall_accuracy:.1%}")
        
        print(f"\n📋 File-by-file breakdown:")
        for file_acc in file_accuracies:
            status = "✅" if file_acc['overall_accuracy'] >= 0.8 else "⚠️" if file_acc['overall_accuracy'] >= 0.6 else "❌"
            print(f"  {status} {file_acc['file']:10} - {file_acc['overall_accuracy']:.1%} (Title: {file_acc['title_accuracy']:.1%}, Outline: {file_acc['outline_accuracy']:.1%})")
        
        # Target assessment
        target_met = avg_overall_accuracy >= 0.8
        print(f"\n🎯 TARGET ASSESSMENT:")
        print(f"   Target: 80% accuracy")
        print(f"   Achieved: {avg_overall_accuracy:.1%}")
        if target_met:
            print("   ✅ TARGET MET!")
        else:
            print(f"   ❌ Need {0.8 - avg_overall_accuracy:.1%} more to reach target")
        
        print(f"\n🔍 DETAILED ANALYSIS:")
        if avg_title_accuracy < 0.8:
            print("   📝 Title extraction needs improvement")
        if avg_outline_accuracy < 0.8:
            print("   📋 Outline extraction needs improvement")

if __name__ == "__main__":
    compare_files()
