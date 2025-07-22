# toc_detection.py
import re
from typing import List, Dict
from .pdf_text import PDFTextUtils

class TOCDetectionUtils:
    """Utilities for Table of Contents detection"""
    
    @staticmethod
    def is_table_of_contents_page(page, blocks: List[Dict]) -> bool:
        """Detect if this page is a Table of Contents"""
        page_text = page.get_text().upper()
        
        # Look for "TABLE OF CONTENTS" heading
        if "TABLE OF CONTENTS" in page_text:
            return True
        
        # Look for patterns indicating TOC structure
        toc_indicators = 0
        
        for block in blocks:
            if block["type"] != 0:
                continue
                
            text = PDFTextUtils.extract_block_text(block).strip()
            
            # Check for typical TOC patterns
            if TOCDetectionUtils.is_toc_entry(text):
                toc_indicators += 1
        
        # If we have many TOC-like entries, it's likely a TOC page
        return toc_indicators >= 3

    @staticmethod
    def is_toc_entry(text: str) -> bool:
        """Check if text looks like a table of contents entry"""
        # Pattern 1: "1. Something 5" or "Chapter 1: Title 10"
        if re.search(r'^(\d+\.|\d+\.\d+\.?|Chapter\s+\d+:?)\s+.+\s+\d+\s*$', text.strip()):
            return True
        
        # Pattern 2: Multiple entries concatenated with page numbers
        if re.search(r'\d+\.\d+\s+[^0-9]+\s+\d+\s+\d+\.\d+', text):
            return True
        
        # Pattern 3: Text ending with just a number (page number)
        if re.search(r'^.+\s+\d{1,3}\s*$', text.strip()) and len(text.strip()) > 10:
            return True
        
        return False

    @staticmethod
    def extract_toc_heading_only(blocks: List[Dict], page_num: int) -> List[Dict]:
        """Extract only the 'Table of Contents' heading from TOC page"""
        headings = []
        
        for block in blocks:
            if block["type"] != 0:
                continue
                
            text = PDFTextUtils.extract_block_text(block).strip()
            
            # Only extract the "Table of Contents" heading itself
            if re.match(r'^(Table of Contents|Contents|TOC)$', text, re.IGNORECASE):
                headings.append({
                    'level': 'H1',
                    'text': text,
                    'page': page_num - 1
                })
        
        return headings