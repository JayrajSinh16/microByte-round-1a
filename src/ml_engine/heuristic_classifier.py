import re
import numpy as np
from typing import Dict, List
from src.shared_utils import (
    DocumentAnalysisUtils, PatternMatchingUtils, TextNormalizationUtils
)

class HeuristicClassifier:
    """Handles heuristic-based classification of text blocks"""
    
    def classify(self, blocks: List[Dict]) -> List[Dict]:
        """Advanced heuristic classification when ML not available"""
        # Calculate document statistics
        doc_stats = self._calculate_document_stats(blocks)
        
        classified_blocks = []
        
        for block in blocks:
            # Skip very long blocks (likely paragraphs)
            if block['word_count'] > 30:
                continue
            
            # Calculate heading probability
            heading_score = self._calculate_heading_score(block, doc_stats)
            
            if heading_score > 0.6:  # Lowered threshold to catch more headings
                level = self._determine_level_contextual(block, doc_stats, classified_blocks)
                classified_blocks.append({
                    'block': block,
                    'is_heading': True,
                    'level': level,
                    'confidence': heading_score
                })
        
        return classified_blocks
    
    def _calculate_document_stats(self, blocks: List[Dict]) -> Dict:
        """Calculate document-wide statistics"""
        return DocumentAnalysisUtils.calculate_document_stats(blocks)
    
    def _calculate_heading_score(self, block: Dict, doc_stats: Dict) -> float:
        """Calculate probability that block is a heading using general typography rules"""
        score = 0.0
        text = block['text'].strip()
        
        # 1. FONT SIZE SCORING (most important - universal indicator)
        font_ratio = block['avg_font_size'] / doc_stats['median_font_size'] if doc_stats['median_font_size'] > 0 else 1
        if font_ratio > 2.0:  # Very large text
            score += 0.6
        elif font_ratio > 1.8:  # Large text  
            score += 0.5
        elif block['avg_font_size'] > doc_stats['font_size_percentiles']['p90']:
            score += 0.4
        elif block['avg_font_size'] > doc_stats['font_size_percentiles']['p75']:
            score += 0.25
        
        # 2. POSITION SCORING (universal layout principles)
        if block['x_normalized'] < 0.2:  # Left aligned
            score += 0.1
        if block['y_normalized'] < 0.15 and block['page_num'] == 0:  # Top of first page
            score += 0.15
        
        # 3. TYPOGRAPHY SCORING (universal formatting conventions)
        if block['is_bold']:
            score += 0.2
        if block['is_uppercase']:
            score += 0.15
        if block['word_count'] < 8:  # Short text more likely to be heading
            score += 0.15
        
        # 4. STRUCTURAL PATTERN SCORING (universal document patterns)
        if block['starts_with_number']:  # Numbered items
            score += 0.2
        if re.match(r'^[IVX]+\.', text):  # Roman numerals
            score += 0.25
        if re.match(r'^(What|How|Why|Where|When|Which|Who)\s+', text, re.IGNORECASE):
            score += 0.15  # Question patterns
        if text.endswith(':') and block['word_count'] <= 4:  # Short labels
            score += 0.1
        
        # 5. NEGATIVE SCORING (universal non-heading indicators)
        if block['word_count'] > 15:  # Too long for heading
            score -= 0.3
        if len(text) > 80:  # Very long text
            score -= 0.4
        if re.match(r'^(www\.|http|@|tel:|phone:)', text, re.IGNORECASE):  # Contact info
            score -= 0.5
        if block['has_punctuation'] and block['word_count'] > 10:  # Paragraph-like
            score -= 0.2
        
        return min(max(score, 0), 1)
    
    def _normalize_extracted_text(self, text: str) -> str:
        """Apply general text normalization patterns for common PDF extraction issues"""
        return TextNormalizationUtils.normalize_extracted_text(text)

    def _get_heading_level_patterns(self):
        """Get general, flexible heading level patterns based on universal document conventions"""
        return PatternMatchingUtils.get_heading_level_patterns()

    def _determine_level_contextual(self, block: Dict, doc_stats: Dict, existing_headings: List[Dict]) -> str:
        """Determine heading level based on features and context using configurable pattern-based approach"""
        text = block['text'].strip()
        
        # Get configurable patterns
        level_patterns = self._get_heading_level_patterns()
        
        # Calculate pattern-based level
        best_level = 'H3'  # Default
        best_score = 0.0
        
        for level_config in level_patterns:
            for pattern, base_score in level_config['patterns']:
                if re.match(pattern, text, re.IGNORECASE):
                    # Adjust score based on font size
                    font_bonus = 0.0
                    threshold = level_config['font_threshold']
                    if threshold in doc_stats['font_size_percentiles']:
                        if block['avg_font_size'] > doc_stats['font_size_percentiles'][threshold]:
                            font_bonus = 0.2
                    
                    total_score = base_score + font_bonus
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_level = level_config['level']
                        break  # Use first matching pattern in each level
        
        # Context-based adjustments
        if len(existing_headings) > 0:
            recent_h1_count = sum(1 for h in existing_headings[-5:] if h.get('level') == 'H1')
            if best_level == 'H1' and recent_h1_count > 2:
                best_level = 'H2'  # Avoid too many H1s
        
        return best_level