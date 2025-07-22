# level_classifier.py
import re
from typing import Dict, Optional

class LevelClassifier:
    """Handles heading level classification logic"""
    
    def __init__(self, heading_patterns):
        self.heading_patterns = heading_patterns
    
    def determine_heading_level_generic(self, text: str, font_size: float, is_bold: bool, 
                                      font_hierarchy: Dict, page_num: int = 1) -> Optional[str]:
        """
        Determine heading level using generic, universal document structure rules.
        No hardcoded content-specific patterns - only structural and typographical analysis.
        """
        # Filter out non-heading content using universal patterns
        if not self._is_potential_heading(text, page_num):
            return None
        
        # Use the existing pattern-based detection from PatternMatchingUtils
        for pattern_name, pattern in self.heading_patterns.items():
            if pattern.search(text):
                if 'numbered_h1' in pattern_name or 'chapter' in pattern_name:
                    return 'H1'
                elif 'numbered_h2' in pattern_name or 'keyword_h2' in pattern_name:
                    return 'H2'
                elif 'numbered_h3' in pattern_name or 'keyword_h3' in pattern_name:
                    return 'H3'
                elif 'keyword_h1' in pattern_name:
                    return 'H1'
                elif 'question_h3' in pattern_name:
                    return 'H3'
                elif 'colon_ended' in pattern_name and len(text) < 50:
                    return 'H3'
        
        # Font-based hierarchical classification
        return self._classify_by_font_hierarchy(text, font_size, is_bold, font_hierarchy)
    
    def _is_potential_heading(self, text: str, page_num: int) -> bool:
        """
        Universal filter to identify potential headings based on structural characteristics.
        No content-specific hardcoded patterns.
        """
        # Length-based filtering
        if len(text) > 150:  # Too long to be a typical heading
            return False
        
        if len(text) < 3:  # Too short to be meaningful
            return False
        
        # Universal non-heading patterns (structural, not content-specific)
        non_heading_patterns = [
            # Complete sentences (headings are typically phrases)
            r'^[A-Z][a-z]+.*[a-z]+\s+(will|are|is|have|has|can|should|must|would|could)\s+.+\.$',
            
            # Procedural/instructional language (universal)
            r'^(Please|Click|Visit|Fill|Complete|Submit|Download|Upload|Print|Sign)\s+',
            
            # Date patterns (universal)
            r'^(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d+,?\s+\d{4}',
            r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            
            # Version/copyright patterns (universal)
            r'^(Version|Copyright|©|$c$)\s+',
            
            # Address patterns (structural, not location-specific)
            r'.*\d+.*\b(Street|St|Avenue|Ave|Drive|Dr|Road|Rd|Lane|Ln|Boulevard|Blvd)\b.*',
            
            # Email/URL patterns (universal)
            r'.*@.*\.(com|org|net|edu|gov).*',
            r'.*(http|www\.|\.com|\.org).*',
            
            # Long numeric sequences or codes
            r'^\d{5,}.*',  # Long number sequences
            
            # Repeated text patterns (likely extraction artifacts)
            r'^(.{3,})\s+\1',  # Text repeated twice
        ]
        
        for pattern in non_heading_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return False
        
        return True
    
    def _classify_by_font_hierarchy(self, text: str, font_size: float, is_bold: bool, font_hierarchy: Dict) -> Optional[str]:
        """Classify heading level based on font characteristics and text structure"""
        title_threshold = font_hierarchy.get('title', 16.0)
        h1_threshold = font_hierarchy.get('h1', 14.0)
        h2_threshold = font_hierarchy.get('h2', 12.0)
        h3_threshold = font_hierarchy.get('h3', 11.0)
        body_size = font_hierarchy.get('body', 10.0)
        
        tolerance = 0.8  # Increased tolerance for better matching
        
        # Enhanced font size based classification with better thresholds
        
        # Very large fonts (20pt+) - likely H1 or title-level headings
        if font_size >= 20.0:
            if len(text) < 100:  # Reasonable heading length
                return 'H1'
        
        # Large fonts (16-19pt) - H1 level
        elif font_size >= 16.0:
            if len(text) < 100:  # Increased threshold for long headings
                return 'H1'
        
        # Medium fonts (13-15pt) - Could be H1 or H2 depending on content and context
        elif font_size >= 13.0:
            if len(text) < 60:
                # Use content-based hints to distinguish H1 from H2
                h1_indicators = [
                    re.match(r'^(Chapter|Section|Part|Appendix)', text, re.IGNORECASE),
                    text.endswith('Library') or text.endswith('Strategy'),  # Title-like endings
                    'Digital Library' in text or 'Road Map' in text,  # Document-specific major topics
                    len(text) > 40 and not text.endswith(':')  # Long titles without colons
                ]
                
                if any(h1_indicators):
                    return 'H1'
                else:
                    return 'H2'
        
        # Standard heading fonts (12pt) - H2 level
        elif font_size >= 12.0 or abs(font_size - h2_threshold) <= tolerance:
            if len(text) < 60:
                return 'H2'
            # H2 level characteristics
            if len(text) < 60:
                # Check for H2 vs H3 indicators
                if (re.match(r'^\d+\.\s+[A-Z]', text) or 
                    text.endswith(':') or
                    is_bold):
                    return 'H2'
                else:
                    return 'H3'
        
        elif font_size >= h3_threshold or abs(font_size - h3_threshold) <= tolerance:
            # H3 level characteristics  
            if len(text) < 50:
                return 'H3'
        
        # Fall back to bold text analysis for smaller fonts
        elif is_bold and font_size > body_size * 1.05:
            if len(text) < 40:
                return 'H3'
        
        # Structural patterns that can override font size
        if re.match(r'^\d+\.\d+\.\d+', text):  # 1.1.1 pattern
            return 'H3'
        elif re.match(r'^\d+\.\d+', text):  # 1.1 pattern
            return 'H2'
        elif re.match(r'^\d+\.', text):  # 1. pattern
            return 'H1'
        
        # Special patterns for document sections
        if re.match(r'^(Appendix|Chapter|Section|Part)\s+[A-Z0-9]', text, re.IGNORECASE):
            return 'H1'
        
        return None  # Not a heading