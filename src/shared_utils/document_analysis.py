# document_analysis.py
import re
import numpy as np
from typing import List, Dict
from .pdf_text import PDFTextUtils

class DocumentAnalysisUtils:
    """Utilities for document structure and content analysis"""
    
    @staticmethod
    def calculate_document_stats(blocks: List[Dict]) -> Dict:
        """Calculate document-wide statistics for font analysis"""
        font_sizes = []
        for block in blocks:
            avg_size = block.get('avg_font_size', 0)
            word_count = block.get('word_count', 1)
            if avg_size > 0:
                font_sizes.extend([avg_size] * word_count)
        
        if not font_sizes:
            return {
                'median_font_size': 12,
                'font_size_percentiles': {'p75': 14, 'p90': 16, 'p95': 18},
                'avg_block_words': 5,
                'total_blocks': len(blocks)
            }
        
        return {
            'median_font_size': np.median(font_sizes),
            'font_size_percentiles': {
                'p75': np.percentile(font_sizes, 75),
                'p90': np.percentile(font_sizes, 90),
                'p95': np.percentile(font_sizes, 95)
            },
            'avg_block_words': np.mean([b.get('word_count', 1) for b in blocks]),
            'total_blocks': len(blocks)
        }

    @staticmethod
    def validate_hierarchy(headings: List[Dict]) -> List[Dict]:
        """Ensure heading hierarchy is consistent across document types"""
        if not headings:
            return headings
        
        validated = []
        last_level = 0
        
        level_map = {'H1': 1, 'H2': 2, 'H3': 3, 'H4': 4}
        
        for heading in headings:
            current_level = level_map.get(heading['level'], 1)
            
            # Don't allow skipping levels (H1 -> H3)
            if current_level > last_level + 1:
                heading['level'] = f"H{last_level + 1}"
                current_level = last_level + 1
            
            validated.append(heading)
            last_level = current_level
        
        return validated

    @staticmethod
    def detect_document_type(doc) -> str:
        """
        Detect document type based on structural characteristics, not content-specific keywords.
        
        This method analyzes document structure, formatting patterns, and layout to determine
        the type of document for specialized processing, making it generalizable across
        different domains and content types.
        """
        # Analyze document structure across first few pages
        structure_analysis = DocumentAnalysisUtils._analyze_document_structure(doc)
        
        # Classify based on structural patterns
        if structure_analysis['is_invitation_like']:
            return 'invitation'
        elif structure_analysis['is_academic_like']:
            return 'academic'  
        elif structure_analysis['is_form_like']:
            return 'form'
        elif structure_analysis['is_report_like']:
            return 'report'
        else:
            return 'general'

    @staticmethod
    def _analyze_document_structure(doc) -> Dict:
        """
        Analyze document structure to classify type based on formatting and layout patterns.
        
        Returns a dictionary with boolean flags for different document characteristics.
        """
        analysis = {
            'is_invitation_like': False,
            'is_academic_like': False,
            'is_form_like': False,
            'is_report_like': False,
            'page_count': len(doc),
            'avg_blocks_per_page': 0,
            'has_numbered_sections': False,
            'has_many_short_lines': False,
            'has_centered_text': False,
            'font_variety': 0
        }
        
        total_blocks = 0
        short_lines = 0
        centered_blocks = 0
        numbered_sections = 0
        font_sizes = set()
        
        # Sample first 3 pages for structural analysis
        sample_pages = min(3, len(doc))
        
        for page_num in range(sample_pages):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            total_blocks += len([b for b in blocks if b["type"] == 0])
            
            for block in blocks:
                if block["type"] != 0:  # Only text blocks
                    continue
                
                text = PDFTextUtils.extract_block_text(block)
                if not text.strip():
                    continue
                
                # Analyze block characteristics
                bbox = block['bbox']
                page_width = page.rect.width
                
                # Check for centered text (invitation characteristic)
                block_center = (bbox[0] + bbox[2]) / 2
                page_center = page_width / 2
                if abs(block_center - page_center) < page_width * 0.15:  # Within 15% of center
                    centered_blocks += 1
                
                # Check for short lines (invitation/form characteristic)
                if len(text.strip()) < 50:
                    short_lines += 1
                
                # Check for numbered sections (academic/report characteristic)
                if re.match(r'^\d+\.?\s+[A-Z]', text.strip()):
                    numbered_sections += 1
                
                # Collect font sizes for variety analysis
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        font_sizes.add(round(span.get("size", 10), 1))
        
        # Calculate metrics
        if sample_pages > 0:
            analysis['avg_blocks_per_page'] = total_blocks / sample_pages
        analysis['font_variety'] = len(font_sizes)
        analysis['has_numbered_sections'] = numbered_sections >= 3
        analysis['has_many_short_lines'] = short_lines > total_blocks * 0.6
        analysis['has_centered_text'] = centered_blocks > total_blocks * 0.3
        
        # Classification logic based on structural patterns
        # Invitation-like: Many short lines, centered text, high font variety
        if (analysis['has_many_short_lines'] and 
            analysis['has_centered_text'] and 
            analysis['font_variety'] >= 4):
            analysis['is_invitation_like'] = True
        
        # Academic-like: Numbered sections, moderate line length, structured
        elif (analysis['has_numbered_sections'] and 
              analysis['avg_blocks_per_page'] > 10 and
              not analysis['has_many_short_lines']):
            analysis['is_academic_like'] = True
        
        # Form-like: Many short lines, low font variety, structured layout
        elif (analysis['has_many_short_lines'] and 
              analysis['font_variety'] <= 3 and
              analysis['avg_blocks_per_page'] > 15):
            analysis['is_form_like'] = True
        
        return analysis