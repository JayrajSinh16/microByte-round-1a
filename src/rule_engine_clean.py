# rule_engine_clean.py
import fitz
import re
from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict
from config import Config
from shared_utils import (
    PDFTextUtils, GeometricUtils, DocumentAnalysisUtils, 
    TableDetectionUtils, TOCDetectionUtils, TextNormalizationUtils,
    FontHierarchyAnalyzer, PatternMatchingUtils
)

class SmartRuleEngine:
    """High-performance rule-based heading extractor"""
    
    def __init__(self):
        self.heading_patterns = PatternMatchingUtils.compile_common_patterns()
        self.font_analyzer = FontHierarchyAnalyzer()
        
    def extract(self, pdf_path: str) -> Dict:
        doc = fitz.open(pdf_path)
        
        # Phase 1: Statistical font analysis
        font_hierarchy = self.font_analyzer.analyze(doc)
        
        # Phase 2: Extract title
        title = self._extract_title(doc, font_hierarchy)
        
        # Phase 3: Extract headings
        headings = self._extract_headings(doc, font_hierarchy)
        
        doc.close()
        
        return {
            "title": title,
            "outline": headings
        }
    
    def _extract_title(self, doc, font_hierarchy: Dict) -> str:
        """Extract document title from first few pages"""
        title_candidates = []
        
        # Check first 3 pages
        for page_num in range(min(3, len(doc))):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if block["type"] == 0:
                    text = self._extract_block_text(block).strip()
                    if not text or len(text) > 200:
                        continue
                    
                    font_size = self._get_block_font_size(block)
                    is_bold = self._is_block_bold(block)
                    
                    # Score based on multiple factors
                    score = 0
                    
                    # Higher score for larger fonts
                    if font_size > font_hierarchy['body'] * 1.2:
                        score += 2
                    
                    # Higher score for bold text
                    if is_bold:
                        score += 1
                    
                    # Higher score for being on first page
                    if page_num == 0:
                        score += 2
                    
                    # Lower score for very long text
                    if len(text) > 100:
                        score -= 1
                    
                    # Higher score for text that looks like a title
                    if any(word.lower() in text.lower() for word in ['application', 'form', 'report', 'document']):
                        score += 1
                    
                    if score >= 2:  # Minimum threshold
                        title_candidates.append({
                            'text': text,
                            'score': score,
                            'page': page_num,
                            'font_size': font_size
                        })
        
        # Select best candidate
        if title_candidates:
            # Sort by score (highest first), then by page (earliest first)
            title_candidates.sort(key=lambda x: (-x['score'], x['page'], -x['font_size']))
            return title_candidates[0]['text']
        
        # Fallback: get first substantial text block from first page
        if len(doc) > 0:
            page = doc[0]
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block["type"] == 0:
                    text = PDFTextUtils.extract_block_text(block).strip()
                    if text and 10 <= len(text) <= 200:
                        return text
        
        return "Untitled Document"
    
    def _extract_headings(self, doc, font_hierarchy: Dict) -> List[Dict]:
        """Extract all headings from document"""
        headings = []
        
        for page_num, page in enumerate(doc):
            page_headings = self._extract_page_headings(
                page, page_num + 1, font_hierarchy
            )
            headings.extend(page_headings)
        
        # Post-process to ensure hierarchy consistency
        headings = DocumentAnalysisUtils.validate_hierarchy(headings)
        
        return headings
    
    def _extract_page_headings(self, page, page_num: int, font_hierarchy: Dict) -> List[Dict]:
        """Extract headings from a single page"""
        headings = []
        blocks = page.get_text("dict")["blocks"]
        
        for block_idx, block in enumerate(blocks):
            if block["type"] != 0:  # Skip non-text blocks
                continue
            
            text = PDFTextUtils.extract_block_text(block).strip()
            if not text or len(text) > 200:  # Skip empty or too long
                continue
            
            font_size = PDFTextUtils.get_block_font_size(block)
            is_bold = PDFTextUtils.is_block_bold(block)
            
            # Determine heading level
            level = self._determine_heading_level(
                text, font_size, is_bold, font_hierarchy
            )
            
            if level:
                headings.append({
                    'level': level,
                    'text': text,
                    'page': page_num
                })
        
        return headings
    
    def _determine_heading_level(self, text: str, font_size: float, is_bold: bool, font_hierarchy: Dict) -> str:
        """Determine the heading level based on font properties and patterns"""
        
        # Pattern-based detection first
        for pattern_name, pattern in self.heading_patterns.items():
            if pattern.search(text):
                if 'numbered_h1' in pattern_name or 'chapter' in pattern_name:
                    return 'H1'
                elif 'numbered_h2' in pattern_name or 'keyword_h2' in pattern_name:
                    return 'H2'
                elif 'numbered_h3' in pattern_name:
                    return 'H3'
                elif 'keyword_h1' in pattern_name:
                    return 'H1'
        
        # Font-based detection
        title_threshold = font_hierarchy.get('title', 16.0)
        h1_threshold = font_hierarchy.get('h1', 14.0)
        h2_threshold = font_hierarchy.get('h2', 12.0)
        h3_threshold = font_hierarchy.get('h3', 11.0)
        body_size = font_hierarchy.get('body', 10.0)
        
        # Size-based classification with tolerance
        tolerance = 0.5
        
        if abs(font_size - title_threshold) <= tolerance and len(text) < 100:
            return 'H1'  # Treat title as H1 in outline
        elif abs(font_size - h1_threshold) <= tolerance or font_size >= h1_threshold:
            return 'H1'
        elif abs(font_size - h2_threshold) <= tolerance or font_size >= h2_threshold:
            return 'H2'
        elif abs(font_size - h3_threshold) <= tolerance or font_size >= h3_threshold:
            return 'H3'
        elif is_bold and font_size > body_size:
            return 'H3'
        
        return None  # Not a heading


