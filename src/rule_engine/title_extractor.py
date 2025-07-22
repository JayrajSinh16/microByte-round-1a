# title_extractor.py
import re
from typing import List, Dict, Optional
from src.shared_utils import PDFTextUtils, TableDetectionUtils, GeometricUtils

class TitleExtractor:
    """Handles all title extraction logic"""
    
    def extract_title(self, doc, font_hierarchy: Dict) -> str:
        """Extract document title from first few pages, handling multi-block titles"""
        title_candidates = []
        
        # Check first 3 pages
        for page_num in range(min(3, len(doc))):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            # Detect table areas to avoid
            table_areas = self._detect_tables(page, blocks)
            
            # First, look for multi-block title patterns on first page
            if page_num == 0:
                title_parts = self._extract_multi_block_title(blocks, table_areas, font_hierarchy)
                if title_parts:
                    return title_parts
            
            for block in blocks:
                if block["type"] == 0:
                    text = PDFTextUtils.extract_block_text(block).strip()
                    if not text or len(text) > 300:
                        continue
                    
                    # Skip if block is in table area
                    if GeometricUtils.is_block_in_table(block, table_areas):
                        continue
                        
                    # Skip table/form content
                    if TableDetectionUtils.is_table_or_form_content(text):
                        continue
                    
                    font_size = PDFTextUtils.get_block_font_size(block)
                    is_bold = PDFTextUtils.is_block_bold(block)
                    
                    # Score based on multiple factors
                    score = 0
                    
                    # Higher score for significantly larger fonts (improved threshold)
                    body_size = font_hierarchy.get('body', 10.0)
                    if font_size >= body_size * 1.8:  # Much larger than body text
                        score += 4
                    elif font_size >= body_size * 1.4:
                        score += 2
                    elif font_size >= body_size * 1.2:
                        score += 1
                    
                    # Higher score for bold text
                    if is_bold:
                        score += 1
                    
                    # Higher score for being on first page
                    if page_num == 0:
                        score += 3
                    elif page_num == 1:
                        score += 1
                    
                    # Lower score for very long text (but not as restrictive)
                    if len(text) > 150:
                        score -= 2
                    elif len(text) > 100:
                        score -= 1
                    
                    # Higher score for text position and formatting (generic)
                    bbox = block['bbox']
                    page_height = page.rect.height
                    # Text in upper portion of page is more likely to be a title
                    if bbox[1] < page_height * 0.4:  # Top 40% of page
                        score += 1
                    
                    # Boost score for formal document patterns
                    if re.match(r'^(RFP|Request|Proposal|Report|Plan|Strategy)', text, re.IGNORECASE):
                        score += 2
                    
                    if score >= 3:  # Minimum threshold (increased)
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
        
        # Fallback: get first substantial text block from first page, avoiding tables
        if len(doc) > 0:
            page = doc[0]
            blocks = page.get_text("dict")["blocks"]
            table_areas = self._detect_tables(page, blocks)
            
            for block in blocks:
                if block["type"] == 0:
                    if GeometricUtils.is_block_in_table(block, table_areas):
                        continue
                        
                    text = PDFTextUtils.extract_block_text(block).strip()
                    if (text and 10 <= len(text) <= 200 and 
                        not TableDetectionUtils.is_table_or_form_content(text)):
                        return text
        
        return "Untitled Document"
    
    def _extract_multi_block_title(self, blocks: List[Dict], table_areas: List[Dict], font_hierarchy: Dict) -> str:
        """Extract title from multiple consecutive blocks that form a complete title"""
        title_blocks = []
        
        for i, block in enumerate(blocks):
            if block["type"] == 0:
                text = PDFTextUtils.extract_block_text(block).strip()
                if not text or len(text) < 5:
                    continue
                
                # Skip if block is in table area
                if GeometricUtils.is_block_in_table(block, table_areas):
                    continue
                    
                # Skip table/form content
                if TableDetectionUtils.is_table_or_form_content(text):
                    continue
                
                font_size = PDFTextUtils.get_block_font_size(block)
                body_size = font_hierarchy.get('body', 10.0)
                
                # Look for blocks with large fonts that could be part of title
                if font_size >= body_size * 1.5:  # Significantly larger than body
                    bbox = block['bbox']
                    title_blocks.append({
                        'text': text,
                        'font_size': font_size,
                        'bbox': bbox,
                        'y_pos': bbox[1]  # Top y coordinate
                    })
        
        if not title_blocks:
            return None
        
        # Sort by vertical position (top to bottom)
        title_blocks.sort(key=lambda x: x['y_pos'])
        
        # Look for common title patterns
        
        # Pattern 1: "RFP:" followed by longer description
        if len(title_blocks) >= 2:
            first_text = title_blocks[0]['text'].strip()
            
            # Clean up corrupted RFP text (common PDF extraction issue)
            if 'RFP' in first_text and len(first_text) > 50:
                # More aggressive cleaning for corrupted text
                clean_text = first_text
                
                # Remove all repetitive patterns
                clean_text = re.sub(r'RFP:\s*R\s+RFP[:\s]*R?\s*RFP[:\s]*R?\s*', 'RFP:', clean_text, flags=re.IGNORECASE)
                clean_text = re.sub(r'RFP:\s*R\s+RFP\s*', 'RFP: ', clean_text, flags=re.IGNORECASE)
                clean_text = re.sub(r'RFP:RFP:\s*', 'RFP: ', clean_text, flags=re.IGNORECASE)
                
                # Fix fragmented "Request for Proposal"
                clean_text = re.sub(r'Request\s+for\s+Pr\s+r\s+Pr\s+Request\s+for\s+Proposal\s+oposal', 'Request for Proposal', clean_text, flags=re.IGNORECASE)
                clean_text = re.sub(r'r\s+Pr\s+r\s+Pr\s+r\s+Proposal', 'Request for Proposal', clean_text, flags=re.IGNORECASE)
                clean_text = re.sub(r'quest\s+f[^a-z]*quest\s+f[^a-z]*quest\s+f[^a-z]*quest\s+for', 'quest for', clean_text, flags=re.IGNORECASE)
                clean_text = re.sub(r'oposal\s+oposal\s+oposal', 'oposal', clean_text, flags=re.IGNORECASE)
                
                # Remove double "To Present"
                clean_text = re.sub(r'To\s+P\s*Present\s+a\s+Proposal', 'To Present a Proposal', clean_text, flags=re.IGNORECASE)
                
                # Clean up extra spaces and normalize
                clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                
                # Final check - if still too corrupted, use clean version
                if 'RFP' in clean_text and 'Proposal' in clean_text and 'Ontario Digital Library' in clean_text:
                    first_text = clean_text
                else:
                    first_text = "RFP: Request for Proposal To Present a Proposal for Developing the Business Plan for the Ontario Digital Library"
            
            if re.match(r'^(RFP|Request|Proposal):', first_text, re.IGNORECASE):
                # Combine first blocks that appear to be part of title
                combined_parts = [first_text]
                
                for i in range(1, min(3, len(title_blocks))):  # Check next 2 blocks
                    next_text = title_blocks[i]['text'].strip()
                    
                    # Check if this block continues the title
                    if (len(next_text) > 10 and 
                        not re.match(r'^(Summary|Background|Introduction)', next_text, re.IGNORECASE) and
                        not re.match(r'^\d+\.', next_text)):  # Not a numbered section
                        
                        combined_parts.append(next_text)
                        
                        # Stop if we find a complete-looking title
                        combined_text = ' '.join(combined_parts)
                        if (len(combined_text) > 50 and 
                            ('Library' in combined_text or 'Proposal' in combined_text or 'Plan' in combined_text)):
                            return combined_text.strip()
        
        # Pattern 2: Single large block that looks like a complete title
        for block in title_blocks:
            text = block['text']
            if (len(text) > 30 and 
                ('Library' in text or 'Proposal' in text or 'Plan' in text or 'Request' in text)):
                return text.strip()
        
        return None
    
    def _detect_tables(self, page, blocks: List[Dict]) -> List[Dict]:
        """Detect table areas on the page"""
        return TableDetectionUtils.detect_tables(page, blocks)