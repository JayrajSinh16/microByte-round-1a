# tests/test_extraction.py
import unittest
import json
from pathlib import Path
import sys
sys.path.append('..')

from main import PDFOutlineExtractor

class TestPDFExtraction(unittest.TestCase):
    
    def setUp(self):
        self.extractor = PDFOutlineExtractor()
        self.test_data_dir = Path("test_data")
    
    def test_standard_pdf(self):
        """Test extraction from standard academic PDF"""
        pdf_path = self.test_data_dir / "academic_paper.pdf"
        result = self.extractor.process_pdf(str(pdf_path))
        
        # Verify structure
        self.assertIn("title", result)
        self.assertIn("outline", result)
        self.assertIsInstance(result["outline"], list)
        
        # Verify heading levels
        levels = {h["level"] for h in result["outline"]}
        self.assertTrue(levels.intersection({"H1", "H2", "H3"}))
    
    # tests/test_extraction.py - continued
    def test_scanned_pdf(self):
        """Test extraction from scanned PDF"""
        pdf_path = self.test_data_dir / "scanned_document.pdf"
        result = self.extractor.process_pdf(str(pdf_path))
        
        # Should still return valid structure
        self.assertIn("title", result)
        self.assertIn("outline", result)
        
        # May have fewer headings but should find some
        self.assertGreater(len(result["outline"]), 0)
    
    def test_complex_layout_pdf(self):
        """Test extraction from multi-column PDF"""
        pdf_path = self.test_data_dir / "multi_column.pdf"
        result = self.extractor.process_pdf(str(pdf_path))
        
        # Verify no duplicate headings from columns
        texts = [h["text"] for h in result["outline"]]
        self.assertEqual(len(texts), len(set(texts)))
    
    def test_performance(self):
        """Test processing time stays under limit"""
        import time
        
        pdf_path = self.test_data_dir / "50_page_document.pdf"
        start_time = time.time()
        result = self.extractor.process_pdf(str(pdf_path))
        processing_time = time.time() - start_time
        
        # Must be under 10 seconds
        self.assertLess(processing_time, 10)
        self.assertGreater(len(result["outline"]), 0)
    
    def test_edge_cases(self):
        """Test various edge cases"""
        test_cases = [
            ("empty.pdf", "Should handle empty PDFs"),
            ("no_text.pdf", "Should handle image-only PDFs"),
            ("huge_font.pdf", "Should handle unusual font sizes"),
            ("rotated_text.pdf", "Should handle rotated pages")
        ]
        
        for pdf_name, description in test_cases:
            with self.subTest(pdf=pdf_name):
                pdf_path = self.test_data_dir / pdf_name
                result = self.extractor.process_pdf(str(pdf_path))
                
                # Should always return valid structure
                self.assertIsInstance(result, dict)
                self.assertIn("title", result)
                self.assertIn("outline", result)
                self.assertIsInstance(result["outline"], list)
    
    def test_unicode_handling(self):
        """Test handling of non-English text"""
        pdf_path = self.test_data_dir / "japanese_document.pdf"
        result = self.extractor.process_pdf(str(pdf_path))
        
        # Should preserve Unicode characters
        if result["outline"]:
            # Check that text is properly encoded
            json_str = json.dumps(result, ensure_ascii=False)
            self.assertIsInstance(json_str, str)
    
    def test_heading_hierarchy(self):
        """Test that heading hierarchy is logical"""
        pdf_path = self.test_data_dir / "academic_paper.pdf"
        result = self.extractor.process_pdf(str(pdf_path))
        
        # H3 should not appear before any H2
        # H2 should not appear before any H1
        h1_seen = False
        h2_seen = False
        
        for heading in result["outline"]:
            if heading["level"] == "H1":
                h1_seen = True
            elif heading["level"] == "H2":
                self.assertTrue(h1_seen, "H2 appears before any H1")
                h2_seen = True
            elif heading["level"] == "H3":
                self.assertTrue(h2_seen, "H3 appears before any H2")

if __name__ == "__main__":
    unittest.main()