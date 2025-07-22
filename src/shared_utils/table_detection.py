# table_detection.py
import re
from typing import List, Dict, Optional
from .pdf_text import PDFTextUtils

class TableDetectionUtils:
    """Utilities for detecting and handling table structures"""
    
    @staticmethod
    def is_table_structure(text: str) -> bool:
        """Check if text looks like table structure"""
        # Common table headers and patterns
        table_patterns = [
            r'S\.?No\.?\s+(Name|Description|Item)',  # S.No Name Age etc
            r'(Name|Item|Description)\s+(Age|Quantity|Amount)',
            r'\d+\.\s+\d+\.\s+\d+\.',  # Multiple numbered items in sequence
            r'(Name|Age|Relationship)\s+(Name|Age|Relationship)',  # Repeated headers
        ]
        
        for pattern in table_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # Check for tabular data patterns
        if re.search(r'S\.?No\.?\s+Name\s+Age\s+Relationship', text, re.IGNORECASE):
            return True
            
        # Multiple numbers separated by periods/spaces (form fields)
        if re.match(r'^\d+\.\s*\d+\.\s*\d+\.\s*\d+', text.strip()):
            return True
            
        return False

    @staticmethod
    def is_table_or_form_content(text: str) -> bool:
        """Check if text is typical table or form content that should be ignored"""
        # Skip single numbers or short numbered items
        if re.match(r'^\d+\.?\s*$', text.strip()):
            return True
            
        # Skip typical form field patterns
        form_patterns = [
            r'^\d+\.\s*(Name|Designation|PAY|Whether|Home Town|Amount)',
            r'^S\.?No\.?\s+(Name|Age|Relationship)',
            r'^\d+\.\s+\d+\.\s*$',  # Two sequential numbers like "5. 6."
            r'^\d+\.\s+\d+\.\s+\d+\.\s*$',  # Three sequential numbers like "1. 2. 3."
            r'^\d+\.\s+\d+\.\s+\d+\.\s+\d+',  # Four or more sequential numbers
            r'^(Date|Signature)',
            r'^Rs\.\s*$',  # Currency symbols
            r'^\d+\s+\d+\s+\d+\s*$',  # Numbers without periods like "1 2 3"
            r'^\d+\s*\d+\s*\d+\s*$',  # Numbers closely spaced like "123"
        ]
        
        for pattern in form_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
                
        # Skip very short texts that are likely labels
        if len(text.strip()) <= 2 and text.strip().isdigit():
            return True
            
        # Skip single letters or very short words in numbered contexts
        if re.match(r'^\d+\.\s*[A-Z]\.?\s*$', text.strip()):
            return True
            
        # Skip short sequences of numbers and periods (form field patterns)
        if re.match(r'^[\d\.\s]{2,10}$', text.strip()) and '.' in text:
            return True
            
        return False

    @staticmethod
    def detect_tables(page, blocks: List[Dict]) -> List[Dict]:
        """Detect table areas on the page"""
        table_areas = []
        
        # Method 1: Look for table-like patterns in text blocks
        for block in blocks:
            if block["type"] != 0:
                continue
                
            text = PDFTextUtils.extract_block_text(block).strip()
            
            # Check if this looks like a table header/structure
            if TableDetectionUtils.is_table_structure(text):
                table_areas.append({
                    'bbox': block['bbox'],
                    'type': 'text_table',
                    'confidence': 0.8
                })
        
        # Method 2: Use PyMuPDF's table detection if available
        try:
            layout_tables = page.find_tables()
            for table in layout_tables:
                table_areas.append({
                    'bbox': table.bbox,
                    'type': 'detected_table', 
                    'confidence': 0.9
                })
        except:
            # Fallback: basic geometric detection
            pass
            
        # Method 3: Detect form-like structures (multiple numbered items)
        form_area = TableDetectionUtils.detect_form_structure(blocks)
        if form_area:
            table_areas.append(form_area)
        
        return table_areas

    @staticmethod
    def detect_form_structure(blocks: List[Dict]) -> Optional[Dict]:
        """Detect form-like structures with numbered fields"""
        numbered_blocks = []
        
        for block in blocks:
            if block["type"] != 0:
                continue
                
            text = PDFTextUtils.extract_block_text(block).strip()
            
            # Look for numbered form fields, but exclude likely headings
            if (re.match(r'^\d+\.\s*(.{0,50})?$', text.strip()) and
                not TableDetectionUtils.is_table_or_form_content(text)):
                numbered_blocks.append(block)
        
        # If we have many numbered items, it's likely a form
        if len(numbered_blocks) >= 8:  # At least 8 numbered items
            # Calculate bounding box for the form area
            min_x = min(block['bbox'][0] for block in numbered_blocks)
            min_y = min(block['bbox'][1] for block in numbered_blocks) 
            max_x = max(block['bbox'][2] for block in numbered_blocks)
            max_y = max(block['bbox'][3] for block in numbered_blocks)
            
            return {
                'bbox': [min_x, min_y, max_x, max_y],
                'type': 'form_structure',
                'confidence': 0.7
            }
        
        return None