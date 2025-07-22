# heading_extractor.py
import re
from typing import List, Dict
from shared_utils import (
    PDFTextUtils, DocumentAnalysisUtils, TableDetectionUtils, 
    TOCDetectionUtils, GeometricUtils
)
from .level_classifier import LevelClassifier

class HeadingExtractor:
    """Handles all heading extraction logic"""
    
    def __init__(self, heading_patterns):
        self.heading_patterns = heading_patterns
        self.level_classifier = LevelClassifier(heading_patterns)
    
    def extract_headings(self, doc, font_hierarchy: Dict, title: str = None) -> List[Dict]:
        """Extract all headings from document, excluding the title"""
        headings = []
        
        # Detect document type for specialized extraction
        doc_type = DocumentAnalysisUtils.detect_document_type(doc)
        
        for page_num, page in enumerate(doc):
            page_headings = self._extract_page_headings(
                page, page_num + 1, font_hierarchy, title, doc_type
            )
            headings.extend(page_headings)
        
        # Post-process to ensure hierarchy consistency
        headings = DocumentAnalysisUtils.validate_hierarchy(headings)
        
        return headings
    
    def _extract_page_headings(self, page, page_num: int, font_hierarchy: Dict, title: str = None, doc_type: str = 'general') -> List[Dict]:
        """Extract headings from a single page using generic structure-based approach"""
        blocks = page.get_text("dict")["blocks"]
        
        # Detect if this is a Table of Contents page
        if TOCDetectionUtils.is_table_of_contents_page(page, blocks):
            # For TOC pages, only extract the "Table of Contents" heading itself
            return TOCDetectionUtils.extract_toc_heading_only(blocks, page_num)
        
        # Use generic structure-based extraction for all document types
        return self._extract_generic_headings(blocks, page_num, font_hierarchy, title)

    def _extract_generic_headings(self, blocks: List[Dict], page_num: int, font_hierarchy: Dict, title: str = None) -> List[Dict]:
        """Generic heading extraction based on universal document structure principles"""
        headings = []
        
        # Detect table areas to avoid
        table_areas = TableDetectionUtils.detect_tables(None, blocks) if blocks else []
        
        for block in blocks:
            if block["type"] != 0:  # Skip non-text blocks
                continue
            
            text = PDFTextUtils.extract_block_text(block).strip()
            if not text:
                continue
            
            # Skip if block is in table area
            if GeometricUtils.is_block_in_table(block, table_areas):
                continue
                
            # Skip typical table/form content patterns
            if TableDetectionUtils.is_table_or_form_content(text):
                continue
                
            # Skip Table of Contents entries
            if TOCDetectionUtils.is_toc_entry(text):
                continue
                
            # Skip if this text matches the title (avoid duplication)
            if title and text.strip().lower() == title.strip().lower():
                continue
            
            # Extract font characteristics
            font_size = PDFTextUtils.get_block_font_size(block)
            is_bold = PDFTextUtils.is_block_bold(block)
            
            # Determine heading level using universal rules
            level = self.level_classifier.determine_heading_level_generic(
                text, font_size, is_bold, font_hierarchy, page_num
            )
            
            if level:
                headings.append({
                    'level': level,
                    'text': text,
                    'page': page_num  # Already 1-indexed from _extract_page_headings
                })
        
        return headings
    
    def _extract_headings_from_long_block(self, text: str, page_num: int) -> List[Dict]:
        """Extract potential headings from long text blocks using generic patterns"""
        headings = []
        
        # Split by common sentence endings and look for heading patterns
        sentences = re.split(r'[.!?]\s+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 5:
                continue
                
            # Use generic heading detection patterns
            if self._looks_like_heading_fragment(sentence):
                headings.append({
                    'level': 'H3',  # Default to H3 for extracted fragments
                    'text': sentence,
                    'page': page_num  # Already 1-indexed
                })
        
        return headings
    
    def _looks_like_heading_fragment(self, text: str) -> bool:
        """Check if a text fragment looks like a heading using generic patterns"""
        # Generic heading characteristics
        if len(text) > 100:  # Too long to be a heading
            return False
            
        # Structural patterns that indicate headings
        heading_indicators = [
            len(text) < 50 and text.isupper(),  # Short all-caps text
            text.endswith(':') and len(text) < 30,  # Short colon-ended text
            re.match(r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$', text) and len(text) < 40,  # Title Case
            re.match(r'^\d+\.?\s+[A-Z]', text),  # Numbered sections
            re.match(r'^[A-Z][a-z]+\s+[IVX]+:', text),  # "Phase I:", "Chapter II:", etc.
            re.match(r'^(What|How|Why|Where|When|Which)\s+', text),  # Question patterns
        ]
        
        return any(heading_indicators)