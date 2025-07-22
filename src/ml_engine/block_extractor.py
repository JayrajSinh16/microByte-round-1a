import re
import numpy as np
from typing import Dict, List
from src.shared_utils import (
    PDFTextUtils, GeometricUtils, TextNormalizationUtils
)

class BlockExtractor:
    """Handles extraction of text blocks with features from PDFs"""
    
    def extract_blocks_with_features(self, page, page_num: int) -> List[Dict]:
        """Extract text blocks with ML-ready features - enhanced for mixed content"""
        blocks = []
        page_dict = page.get_text("dict")
        page_height = page_dict["height"]
        page_width = page_dict["width"]
        
        for block_idx, block in enumerate(page_dict["blocks"]):
            if block["type"] != 0:  # Skip non-text
                continue
            
            # For mixed content PDFs, extract lines separately if they have very different characteristics
            lines = block.get("lines", [])
            if len(lines) > 1:
                # Check if this block contains dramatically different line styles
                should_split = self._should_split_block_lines(lines)
            else:
                should_split = False
            
            if should_split:
                # Extract each line as a separate block
                for line_idx, line in enumerate(lines):
                    line_text = self._extract_text_from_line(line)
                    if not line_text.strip():
                        continue
                    
                    # Create features for this line
                    line_bbox = self._calculate_line_bbox(line)
                    features = self._create_line_features(line, line_text, line_bbox, 
                                                        page_num, block_idx, line_idx,
                                                        page_width, page_height)
                    if features:
                        blocks.append(features)
            else:
                # Extract the entire block as before
                text = self._extract_text_from_block(block)
                if not text.strip():
                    continue
                
                # Extract features
                features = {
                    'text': text,
                    'page_num': page_num,
                    'block_idx': block_idx,
                    
                    # Position features
                    'x_normalized': block["bbox"][0] / page_width,
                    'y_normalized': block["bbox"][1] / page_height,
                    'width_normalized': (block["bbox"][2] - block["bbox"][0]) / page_width,
                    'height_normalized': (block["bbox"][3] - block["bbox"][1]) / page_height,
                    
                    # Text features
                    'text_length': len(text),
                    'word_count': len(text.split()),
                    'is_uppercase': text.isupper(),
                    'starts_with_number': bool(re.match(r'^\d+', text)),
                    'has_punctuation': text.strip()[-1] in '.!?:' if text.strip() else False,
                    'ends_with_colon': text.strip().endswith(':'),
                    
                    # Pattern features
                    'is_question': bool(re.match(r'^(What|How|Why|Where|When)', text, re.IGNORECASE)),
                    'is_numbered_list': bool(re.match(r'^\d+\.', text)),
                    'is_alphabetic_list': bool(re.match(r'^[a-zA-Z]\.', text)),
                    'has_chapter_keyword': bool(re.search(r'\b(chapter|section|part|appendix)\b', text, re.IGNORECASE)),
                    
                    # Font features
                    'font_sizes': [],
                    'is_bold': False,
                    'is_italic': False,
                    'font_names': set()
                }
                
                # Extract font information
                self._extract_font_features(block, features)
                
                # Calculate font statistics (matching training data)
                if features['font_sizes']:
                    features['avg_font_size'] = np.mean(features['font_sizes'])
                    features['max_font_size'] = max(features['font_sizes'])
                    features['min_font_size'] = min(features['font_sizes'])
                    features['font_size_variance'] = np.var(features['font_sizes'])
                else:
                    features['avg_font_size'] = 0
                    features['max_font_size'] = 0
                    features['min_font_size'] = 0
                    features['font_size_variance'] = 0
                
                features['font_variety'] = len(features['font_names'])
                
                # Relative font size features (simplified since we don't have font hierarchy here)
                features['font_ratio_to_body'] = features['avg_font_size'] / 12 if features['avg_font_size'] > 0 else 1
                features['is_larger_than_body'] = features['avg_font_size'] > 12
                
                blocks.append(features)
        
        return blocks
    
    def _should_split_block_lines(self, lines: List[Dict]) -> bool:
        """Determine if block lines should be split due to different characteristics"""
        if len(lines) < 2:
            return False
        
        # Calculate font size statistics for all spans
        all_font_sizes = []
        line_avg_sizes = []
        
        for line in lines:
            line_sizes = []
            for span in line.get("spans", []):
                size = span.get("size", 0)
                all_font_sizes.append(size)
                line_sizes.append(size)
            
            if line_sizes:
                line_avg_sizes.append(np.mean(line_sizes))
        
        if len(line_avg_sizes) < 2:
            return False
        
        # Split if there's a significant difference in font sizes between lines
        max_line_size = max(line_avg_sizes)
        min_line_size = min(line_avg_sizes)
        
        # Split if font size range is > 10px and ratio is > 1.5
        if (max_line_size - min_line_size) > 10 and (max_line_size / min_line_size) > 1.5:
            return True
        
        return False
    
    def _calculate_line_bbox(self, line: Dict) -> List[float]:
        """Calculate bounding box for a line"""
        return GeometricUtils.calculate_line_bbox(line)
    
    def _extract_text_from_line(self, line: Dict) -> str:
        """Extract text from a single line with improved spacing"""
        return TextNormalizationUtils.extract_text_from_line(line)
    
    def _create_line_features(self, line: Dict, text: str, bbox: List[float], 
                            page_num: int, block_idx: int, line_idx: int,
                            page_width: float, page_height: float) -> Dict:
        """Create features for an individual line"""
        if not text.strip():
            return None
        
        features = {
            'text': text,
            'page_num': page_num,
            'block_idx': block_idx,
            'line_idx': line_idx,
            
            # Position features
            'x_normalized': bbox[0] / page_width,
            'y_normalized': bbox[1] / page_height,
            'width_normalized': (bbox[2] - bbox[0]) / page_width,
            'height_normalized': (bbox[3] - bbox[1]) / page_height,
            
            # Text features
            'text_length': len(text),
            'word_count': len(text.split()),
            'is_uppercase': text.isupper(),
            'starts_with_number': bool(re.match(r'^\d+', text)),
            'has_punctuation': text.strip()[-1] in '.!?',
            
            # Font features
            'font_sizes': [],
            'is_bold': False,
            'is_italic': False,
            'font_names': set()
        }
        
        # Extract font information from this line
        font_features = PDFTextUtils.extract_font_features({'lines': [line]})
        features.update(font_features)
        
        return features
    
    def _extract_font_features(self, block: Dict, features: Dict):
        """Extract font-related features from block"""
        font_info = PDFTextUtils.extract_font_features(block)
        features.update(font_info)
    
    def _extract_text_from_block(self, block: Dict) -> str:
        """Extract text from a PyMuPDF block"""
        return PDFTextUtils.extract_block_text(block)
    
    def extract_text_from_block(self, block: Dict) -> str:
        """Public method to extract text from block"""
        return self._extract_text_from_block(block)