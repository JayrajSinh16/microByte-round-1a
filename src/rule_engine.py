# rule_engine.py
import fitz
import re
from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict
from config import Config

class SmartRuleEngine:
    """High-performance rule-based heading extractor"""
    
    def __init__(self):
        self.heading_patterns = self._compile_patterns()
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
    
    def _compile_patterns(self) -> Dict:
        return {
            'numbered_h1': re.compile(r'^(\d+\.?)\s+(.+)$'),
            'numbered_h2': re.compile(r'^(\d+\.\d+\.?)\s+(.+)$'),
            'numbered_h3': re.compile(r'^(\d+\.\d+\.\d+\.?)\s+(.+)$'),
            'chapter': re.compile(r'^(Chapter|CHAPTER|Section|SECTION)\s+(\d+|[IVX]+)', re.I),
            'keyword_h1': re.compile(r'^(Introduction|Conclusion|Abstract|References|Appendix)', re.I),
            'keyword_h2': re.compile(r'^(Background|Methodology|Results|Discussion|Related Work)', re.I)
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
                    text = self._extract_block_text(block).strip()
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
        headings = self._validate_hierarchy(headings)
        
        return headings
    
    def _extract_page_headings(self, page, page_num: int, font_hierarchy: Dict) -> List[Dict]:
        """Extract headings from a single page"""
        headings = []
        blocks = page.get_text("dict")["blocks"]
        
        for block_idx, block in enumerate(blocks):
            if block["type"] != 0:  # Skip non-text blocks
                continue
            
            text = self._extract_block_text(block).strip()
            if not text or len(text) > 200:  # Skip empty or too long
                continue
            
            font_size = self._get_block_font_size(block)
            is_bold = self._is_block_bold(block)
            
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
    
    def _extract_block_text(self, block) -> str:
        """Extract text from a block"""
        text_parts = []
        for line in block.get("lines", []):
            line_text = ""
            for span in line.get("spans", []):
                line_text += span.get("text", "")
            text_parts.append(line_text.strip())
        return " ".join(text_parts).strip()
    
    def _get_block_font_size(self, block) -> float:
        """Get average font size for a block"""
        sizes = []
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                sizes.append(span.get("size", 10.0))
        return sum(sizes) / len(sizes) if sizes else 10.0
    
    def _is_block_bold(self, block) -> bool:
        """Check if block text is bold"""
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if span.get("flags", 0) & 16:  # Bold flag
                    return True
        return False
    
    def _validate_hierarchy(self, headings: List[Dict]) -> List[Dict]:
        """Ensure heading hierarchy is consistent"""
        if not headings:
            return headings
        
        # Simple validation - ensure we don't jump levels
        validated = []
        last_level = 0
        
        level_map = {'H1': 1, 'H2': 2, 'H3': 3}
        
        for heading in headings:
            current_level = level_map.get(heading['level'], 1)
            
            # Don't allow skipping levels (H1 -> H3)
            if current_level > last_level + 1:
                heading['level'] = f"H{last_level + 1}"
                current_level = last_level + 1
            
            validated.append(heading)
            last_level = current_level
        
        return validated


class FontHierarchyAnalyzer:
    """Statistical analysis of font usage in document"""
    
    def analyze(self, doc) -> Dict:
        font_stats = defaultdict(lambda: {
            'count': 0, 
            'total_chars': 0, 
            'pages': set(),
            'is_bold': False,
            'sample_texts': []
        })
        
        # Collect font statistics
        for page_num, page in enumerate(doc):
            self._analyze_page_fonts(page, page_num, font_stats)
        
        # Determine hierarchy
        hierarchy = self._determine_hierarchy(font_stats)
        return hierarchy
    
    def _analyze_page_fonts(self, page, page_num: int, font_stats: Dict):
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            if block["type"] == 0:  # Text block
                for line in block["lines"]:
                    for span in line["spans"]:
                        font_key = (
                            round(span["size"], 1),
                            span["flags"]  # Bold, italic info
                        )
                        
                        font_stats[font_key]['count'] += 1
                        font_stats[font_key]['total_chars'] += len(span["text"])
                        font_stats[font_key]['pages'].add(page_num)
                        font_stats[font_key]['is_bold'] = bool(span["flags"] & 16)
                        
                        # Store samples for pattern analysis
                        if len(font_stats[font_key]['sample_texts']) < 10:
                            font_stats[font_key]['sample_texts'].append(span["text"].strip())
    
    def _determine_hierarchy(self, font_stats: Dict) -> Dict:
        # Sort fonts by size
        sorted_fonts = sorted(
            font_stats.items(), 
            key=lambda x: x[0][0], 
            reverse=True
        )
        
        if not sorted_fonts:
            return {
                'title': 16.0,
                'h1': 14.0,
                'h2': 12.0,
                'h3': 11.0,
                'body': 10.0
            }
        
        # Filter out body text (most common)
        body_font = max(font_stats.items(), key=lambda x: x[1]['total_chars'])
        body_size = body_font[0][0]  # First element is font size
        
        hierarchy = {
            'title': None,
            'h1': None,
            'h2': None,
            'h3': None,
            'body': body_size
        }
        
        # Assign hierarchy based on size and usage
        significant_fonts = [
            (font, stats) for font, stats in sorted_fonts 
            if stats['count'] > 3 and font[0] > body_size
        ]
        
        if significant_fonts:
            # Title: Largest font, usually on first pages
            title_candidate = significant_fonts[0]
            if (0 in title_candidate[1]['pages'] or 1 in title_candidate[1]['pages']):
                hierarchy['title'] = title_candidate[0][0]
            
            # Assign H1, H2, H3 based on size gaps
            remaining = [f for f in significant_fonts if f[0][0] != hierarchy['title']]
            
            if len(remaining) > 0:
                hierarchy['h1'] = remaining[0][0][0]
            if len(remaining) > 1:
                hierarchy['h2'] = remaining[1][0][0]
            if len(remaining) > 2:
                hierarchy['h3'] = remaining[2][0][0]
        
        # Fill in missing values with defaults
        if hierarchy['title'] is None:
            hierarchy['title'] = body_size * 1.5
        if hierarchy['h1'] is None:
            hierarchy['h1'] = body_size * 1.3
        if hierarchy['h2'] is None:
            hierarchy['h2'] = body_size * 1.15
        if hierarchy['h3'] is None:
            hierarchy['h3'] = body_size * 1.1
        
        return hierarchy
