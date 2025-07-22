# text_normalization.py
import re
from typing import Dict

class TextNormalizationUtils:
    """Utilities for normalizing and cleaning extracted text"""
    
    @staticmethod
    def normalize_extracted_text(text: str) -> str:
        """Apply general text normalization patterns for common PDF extraction issues"""
        
        # Universal PDF text extraction fixes (not document-specific)
        normalization_patterns = [
            # Fix broken word spacing (common OCR issue)
            (r'([a-z])([A-Z])', r'\1 \2'),  # Add space between lowercase-uppercase
            (r'([A-Z])([A-Z][a-z])', r'\1 \2'),  # Fix consecutive capitals
            
            # Fix common spacing issues
            (r'\s+', ' '),  # Normalize multiple spaces
            (r'^\s+|\s+$', ''),  # Trim whitespace
            
            # Fix punctuation spacing (universal)
            (r'([.!?])\s*([A-Z])', r'\1 \2'),  # Space after sentence endings
            (r'\s*([!?:;])\s*', r'\1 '),  # Normalize punctuation spacing
        ]
        
        result = text
        for pattern, replacement in normalization_patterns:
            result = re.sub(pattern, replacement, result)
        
        return result.strip()

    @staticmethod
    def extract_text_from_line(line: Dict) -> str:
        """Extract text from a single line with improved spacing"""
        text_parts = []
        for span in line.get("spans", []):
            span_text = span.get("text", "")
            if span_text:
                text_parts.append(span_text)
        
        # Join all text first
        result = "".join(text_parts)
        
        # Apply intelligent text normalization patterns
        result = TextNormalizationUtils.normalize_extracted_text(result)
        
        return result