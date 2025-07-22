# pdf_text.py
import numpy as np
from typing import Dict

class PDFTextUtils:
    """Utilities for extracting and processing text from PDF blocks"""
    
    @staticmethod
    def extract_block_text(block: Dict) -> str:
        """Extract text from a PyMuPDF block"""
        text_parts = []
        for line in block.get("lines", []):
            line_text = ""
            for span in line.get("spans", []):
                span_text = span.get("text", "").strip()
                if span_text:
                    line_text += span_text + " "
            if line_text.strip():
                text_parts.append(line_text.strip())
        return " ".join(text_parts)

    @staticmethod
    def get_block_font_size(block: Dict) -> float:
        """Get average font size for a block"""
        sizes = []
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                sizes.append(span.get("size", 10.0))
        return sum(sizes) / len(sizes) if sizes else 10.0

    @staticmethod
    def is_block_bold(block: Dict) -> bool:
        """Check if block text is bold"""
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                flags = span.get("flags", 0)
                if flags & 2**4:  # Bold flag
                    return True
        return False

    @staticmethod
    def extract_font_features(block: Dict) -> Dict:
        """Extract comprehensive font features from a block"""
        features = {
            'font_sizes': [],
            'font_names': set(),
            'is_bold': False,
            'is_italic': False
        }
        
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                features['font_sizes'].append(span.get("size", 0))
                features['font_names'].add(span.get("font", ""))
                
                flags = span.get("flags", 0)
                if flags & 2**4:  # Bold
                    features['is_bold'] = True
                if flags & 2**1:  # Italic
                    features['is_italic'] = True
        
        # Calculate aggregates
        if features['font_sizes']:
            features['avg_font_size'] = np.mean(features['font_sizes'])
            features['max_font_size'] = max(features['font_sizes'])
            features['min_font_size'] = min(features['font_sizes'])
        else:
            features['avg_font_size'] = 0
            features['max_font_size'] = 0 
            features['min_font_size'] = 0
        
        features['font_variety'] = len(features['font_names'])
        
        return features