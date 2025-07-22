# pattern_matching.py
import re
from typing import Dict

class PatternMatchingUtils:
    """Utilities for pattern-based heading detection"""
    
    @staticmethod
    def compile_common_patterns() -> Dict:
        """Compile common heading patterns used across engines"""
        return {
            'numbered_h1': re.compile(r'^(\d+\.?)\s+(.+)$'),
            'numbered_h2': re.compile(r'^(\d+\.\d+\.?)\s+(.+)$'),
            'numbered_h3': re.compile(r'^(\d+\.\d+\.\d+\.?)\s+(.+)$'),
            'chapter': re.compile(r'^(Chapter|CHAPTER|Section|SECTION)\s+(\d+|[IVX]+)', re.I),
            'keyword_h1': re.compile(r'^(Introduction|Conclusion|Abstract|References|Appendix)', re.I),
            'keyword_h2': re.compile(r'^(Background|Methodology|Results|Discussion|Related Work|Summary)', re.I),
            'keyword_h3': re.compile(r'^(Timeline|Milestones|Approach)', re.I),
            'question_h3': re.compile(r'^(What|How|Why|Where|When).*\?', re.I),
            'colon_ended': re.compile(r'^.+:$'),
            'ontario_subsection': re.compile(r'^For (each|the) Ontario', re.I)
        }

    @staticmethod
    def get_heading_level_patterns():
        """
        Get general, flexible heading level patterns based on universal document conventions
        
        DESIGN PHILOSOPHY:
        - Focus on STRUCTURE and TYPOGRAPHY, not content
        - Use universal formatting conventions (font size, numbering, case)
        - Avoid document-specific keywords or phrases
        - Prioritize maintainability over marginal accuracy gains
        """
        return [
            # H1 patterns - Major structural elements
            {
                'level': 'H1',
                'patterns': [
                    # Structural patterns (not content-specific)
                    (r'^[A-Z][A-Z\s]{8,}[!?]?\s*$', 0.8),  # Long all-caps text (8+ chars)
                    (r'^(CHAPTER|PART|SECTION)\s+\d+', 0.7),  # Numbered major divisions
                    (r'^[IVX]+\.\s+', 0.7),  # Roman numerals
                ],
                'font_threshold': 'p95'  # Must be in top 5% font sizes
            },
            
            # H2 patterns - Main sections
            {
                'level': 'H2', 
                'patterns': [
                    # Numbered sections
                    (r'^\d+\.\s+[A-Z]', 0.7),  # "1. Introduction"
                    (r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*\s*$', 0.6),  # Title Case Words
                    
                    # All caps but shorter than H1
                    (r'^[A-Z\s]{4,8}\s*$', 0.5),  # Short all-caps (4-8 chars)
                ],
                'font_threshold': 'p90'  # Top 10% font sizes
            },
            
            # H3 patterns - Subsections
            {
                'level': 'H3',
                'patterns': [
                    # Question patterns (universal)
                    (r'^(What|How|Why|Where|When|Which|Who)\s+', 0.6),
                    
                    # Sub-numbered items
                    (r'^\d+\.\d+', 0.6),  # "1.1 Subsection"
                    (r'^[a-z]\)', 0.5),   # "a) item"
                    
                    # Colon-ended labels
                    (r'^[A-Z][a-zA-Z\s]{2,20}:\s*$', 0.4),  # "Objective:" 
                ],
                'font_threshold': 'p75'  # Top 25% font sizes
            },
            
            # H4 patterns - Minor subsections
            {
                'level': 'H4',
                'patterns': [
                    # Detailed numbering
                    (r'^\d+\.\d+\.\d+', 0.6),  # "1.1.1 Detail"
                    (r'^[a-z]\.', 0.5),        # "a. item"
                    (r'^$[a-z]$', 0.4),      # "(a) item"
                ],
                'font_threshold': 'p50'  # Median font size
            }
        ]