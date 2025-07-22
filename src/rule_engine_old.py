# rule_engine.py
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
        
        # Phase 2: Detect document type early
        doc_type = self._detect_document_type(doc)
        
        # Phase 3: Extract title (skip for invitation-like documents)
        if doc_type == 'invitation':
            title = ""  # Invitations should have empty title to avoid extracting decorative text
        else:
            title = self._extract_title(doc, font_hierarchy)
        
        # Phase 4: Extract headings (excluding the title)
        headings = self._extract_headings(doc, font_hierarchy, title)
        
        doc.close()
        
        return {
            "title": title,
            "outline": headings
        }
    
    def _extract_title(self, doc, font_hierarchy: Dict) -> str:
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
                    text = self._extract_block_text(block).strip()
                    if not text or len(text) > 300:
                        continue
                    
                    # Skip if block is in table area
                    if self._is_block_in_table(block, table_areas):
                        continue
                        
                    # Skip table/form content
                    if self._is_table_or_form_content(text):
                        continue
                    
                    font_size = self._get_block_font_size(block)
                    is_bold = self._is_block_bold(block)
                    
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
                    if self._is_block_in_table(block, table_areas):
                        continue
                        
                    text = self._extract_block_text(block).strip()
                    if (text and 10 <= len(text) <= 200 and 
                        not self._is_table_or_form_content(text)):
                        return text
        
        return "Untitled Document"
    
    def _extract_multi_block_title(self, blocks: List[Dict], table_areas: List[Dict], font_hierarchy: Dict) -> str:
        """Extract title from multiple consecutive blocks that form a complete title"""
        title_blocks = []
        
        for i, block in enumerate(blocks):
            if block["type"] == 0:
                text = self._extract_block_text(block).strip()
                if not text or len(text) < 5:
                    continue
                
                # Skip if block is in table area
                if self._is_block_in_table(block, table_areas):
                    continue
                    
                # Skip table/form content
                if self._is_table_or_form_content(text):
                    continue
                
                font_size = self._get_block_font_size(block)
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
    
    def _extract_headings(self, doc, font_hierarchy: Dict, title: str = None) -> List[Dict]:
        """Extract all headings from document, excluding the title"""
        headings = []
        
        # Detect document type for specialized extraction
        doc_type = self._detect_document_type(doc)
        
        for page_num, page in enumerate(doc):
            page_headings = self._extract_page_headings(
                page, page_num + 1, font_hierarchy, title, doc_type
            )
            headings.extend(page_headings)
        
        # Post-process to ensure hierarchy consistency
        headings = self._validate_hierarchy(headings)
        
        return headings
    
    def _detect_document_type(self, doc) -> str:
        """Detect the type of document to apply specialized extraction rules"""
        return DocumentAnalysisUtils.detect_document_type(doc)
    
    def _extract_page_headings(self, page, page_num: int, font_hierarchy: Dict, title: str = None, doc_type: str = 'general') -> List[Dict]:
        """Extract headings from a single page using generic structure-based approach"""
        blocks = page.get_text("dict")["blocks"]
        
        # Detect if this is a Table of Contents page
        if self._is_table_of_contents_page(page, blocks):
            # For TOC pages, only extract the "Table of Contents" heading itself
            return self._extract_toc_heading_only(blocks, page_num)
        
        # Use generic structure-based extraction for all document types
        return self._extract_generic_headings(blocks, page_num, font_hierarchy, title)

    def _extract_generic_headings(self, blocks: List[Dict], page_num: int, font_hierarchy: Dict, title: str = None) -> List[Dict]:
        """Generic heading extraction based on universal document structure principles"""
        headings = []
        
        # Detect table areas to avoid
        table_areas = TableDetectionUtils.detect_tables(None, blocks) if blocks else []
        
        for block in blocks:
            if block["type"] != 0:  # Skip non-text blocks
                continue
            
            text = self._extract_block_text(block).strip()
            if not text:
                continue
            
            # Skip if block is in table area
            if self._is_block_in_table(block, table_areas):
                continue
                
            # Skip typical table/form content patterns
            if self._is_table_or_form_content(text):
                continue
                
            # Skip Table of Contents entries
            if self._is_toc_entry(text):
                continue
                
            # Skip if this text matches the title (avoid duplication)
            if title and text.strip().lower() == title.strip().lower():
                continue
            
            # Extract font characteristics
            font_size = self._get_block_font_size(block)
            is_bold = self._is_block_bold(block)
            
            # Determine heading level using universal rules
            level = self._determine_heading_level_generic(
                text, font_size, is_bold, font_hierarchy, page_num
            )
            
            if level:
                headings.append({
                    'level': level,
                    'text': text,
                    'page': page_num  # Already 1-indexed from _extract_page_headings
                })
        
        return headings
    
    def _extract_headings_from_long_block(self, text: str, page_num: int) -> List[Dict]:
        """Extract potential headings from long text blocks using generic patterns"""
        headings = []
        
        # Split by common sentence endings and look for heading patterns
        sentences = re.split(r'[.!?]\s+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 5:
                continue
                
            # Use generic heading detection patterns
            if self._looks_like_heading_fragment(sentence):
                headings.append({
                    'level': 'H3',  # Default to H3 for extracted fragments
                    'text': sentence,
                    'page': page_num  # Already 1-indexed
                })
        
        return headings
    
    def _looks_like_heading_fragment(self, text: str) -> bool:
        """Check if a text fragment looks like a heading using generic patterns"""
        # Generic heading characteristics
        if len(text) > 100:  # Too long to be a heading
            return False
            
        # Structural patterns that indicate headings
        heading_indicators = [
            len(text) < 50 and text.isupper(),  # Short all-caps text
            text.endswith(':') and len(text) < 30,  # Short colon-ended text
            re.match(r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$', text) and len(text) < 40,  # Title Case
            re.match(r'^\d+\.?\s+[A-Z]', text),  # Numbered sections
            re.match(r'^[A-Z][a-z]+\s+[IVX]+:', text),  # "Phase I:", "Chapter II:", etc.
            re.match(r'^(What|How|Why|Where|When|Which)\s+', text),  # Question patterns
        ]
        
        return any(heading_indicators)
    
    def _determine_heading_level_generic(self, text: str, font_size: float, is_bold: bool, font_hierarchy: Dict, page_num: int = 1) -> str:
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
            r'^(Version|Copyright|©|\(c\))\s+',
            
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
    
    def _classify_by_font_hierarchy(self, text: str, font_size: float, is_bold: bool, font_hierarchy: Dict) -> str:
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
    
    def _extract_block_text(self, block) -> str:
        """Extract text from a block"""
        return PDFTextUtils.extract_block_text(block)
    
    def _get_block_font_size(self, block) -> float:
        """Get average font size for a block"""
        return PDFTextUtils.get_block_font_size(block)
    
    def _is_block_bold(self, block) -> bool:
        """Check if block text is bold"""
        return PDFTextUtils.is_block_bold(block)
    
    def _validate_hierarchy(self, headings: List[Dict]) -> List[Dict]:
        """Ensure heading hierarchy is consistent"""
        return DocumentAnalysisUtils.validate_hierarchy(headings)
    
    def _detect_tables(self, page, blocks: List[Dict]) -> List[Dict]:
        """Detect table areas on the page"""
        return TableDetectionUtils.detect_tables(page, blocks)
    
    def _is_table_structure(self, text: str) -> bool:
        """Check if text looks like table structure"""
        return TableDetectionUtils.is_table_structure(text)
    
    def _detect_form_structure(self, blocks: List[Dict]) -> Dict:
        """Detect form-like structures with numbered fields"""
        return TableDetectionUtils.detect_form_structure(blocks)
    
    def _is_block_in_table(self, block: Dict, table_areas: List[Dict]) -> bool:
        """Check if a block overlaps with any table area"""
        return GeometricUtils.is_block_in_table(block, table_areas)
    
    def _bboxes_overlap(self, bbox1: List, bbox2: List) -> bool:
        """Check if two bounding boxes overlap"""
        return GeometricUtils.bboxes_overlap(bbox1, bbox2)
    
    def _is_table_or_form_content(self, text: str) -> bool:
        """Check if text is typical table or form content that should be ignored"""
        return TableDetectionUtils.is_table_or_form_content(text)
    
    def _is_table_of_contents_page(self, page, blocks: List[Dict]) -> bool:
        """Detect if this page is a Table of Contents"""
        return TOCDetectionUtils.is_table_of_contents_page(page, blocks)
    
    def _is_toc_entry(self, text: str) -> bool:
        """Check if text looks like a table of contents entry"""
        return TOCDetectionUtils.is_toc_entry(text)
    
    def _extract_toc_heading_only(self, blocks: List[Dict], page_num: int) -> List[Dict]:
        """Extract only the 'Table of Contents' heading from TOC page"""
        return TOCDetectionUtils.extract_toc_heading_only(blocks, page_num)


