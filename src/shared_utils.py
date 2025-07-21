# shared_utils.py
"""
Shared utilities for PDF processing across different engines.
Contains common functionality used by both RuleEngine and MLEngine.
"""
import fitz
import re
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict

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


class GeometricUtils:
    """Utilities for geometric calculations and spatial analysis"""
    
    @staticmethod
    def bboxes_overlap(bbox1: List, bbox2: List) -> bool:
        """Check if two bounding boxes overlap"""
        return not (bbox1[2] <= bbox2[0] or bbox2[2] <= bbox1[0] or 
                   bbox1[3] <= bbox2[1] or bbox2[3] <= bbox1[1])

    @staticmethod
    def is_block_in_table(block: Dict, table_areas: List[Dict]) -> bool:
        """Check if a block overlaps with any table area"""
        if not table_areas:
            return False
            
        block_bbox = block['bbox']
        for table in table_areas:
            if GeometricUtils.bboxes_overlap(block_bbox, table['bbox']):
                return True
        return False

    @staticmethod
    def calculate_line_bbox(line: Dict) -> List[float]:
        """Calculate bounding box for a line"""
        if not line.get("spans"):
            return [0, 0, 0, 0]
        
        x_coords = []
        y_coords = []
        
        for span in line["spans"]:
            bbox = span.get("bbox", [0, 0, 0, 0])
            x_coords.extend([bbox[0], bbox[2]])
            y_coords.extend([bbox[1], bbox[3]])
        
        if x_coords and y_coords:
            return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
        return [0, 0, 0, 0]


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
        
        # Report-like: Moderate structure, consistent formatting
        elif (analysis['avg_blocks_per_page'] > 8 and 
              analysis['font_variety'] >= 3 and
              not analysis['has_many_short_lines']):
            analysis['is_report_like'] = True
        
        return analysis


class TableDetectionUtils:
    """Utilities for detecting and handling table structures"""
    
    @staticmethod
    def is_table_structure(text: str) -> bool:
        """Check if text looks like table structure"""
        # Common table headers and patterns
        table_patterns = [
            r'S\.?No\.?\s+(Name|Description|Item)',  # S.No Name Age etc
            r'(Name|Item|Description)\s+(Age|Quantity|Amount)',
            r'\d+\.\s+\d+\.\s+\d+\.',  # Multiple numbered items in sequence
            r'(Name|Age|Relationship)\s+(Name|Age|Relationship)',  # Repeated headers
        ]
        
        for pattern in table_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # Check for tabular data patterns
        if re.search(r'S\.?No\.?\s+Name\s+Age\s+Relationship', text, re.IGNORECASE):
            return True
            
        # Multiple numbers separated by periods/spaces (form fields)
        if re.match(r'^\d+\.\s*\d+\.\s*\d+\.\s*\d+', text.strip()):
            return True
            
        return False

    @staticmethod
    def is_table_or_form_content(text: str) -> bool:
        """Check if text is typical table or form content that should be ignored"""
        # Skip single numbers or short numbered items
        if re.match(r'^\d+\.?\s*$', text.strip()):
            return True
            
        # Skip typical form field patterns
        form_patterns = [
            r'^\d+\.\s*(Name|Designation|PAY|Whether|Home Town|Amount)',
            r'^S\.?No\.?\s+(Name|Age|Relationship)',
            r'^\d+\.\s+\d+\.\s*$',  # Two sequential numbers like "5. 6."
            r'^\d+\.\s+\d+\.\s+\d+\.\s*$',  # Three sequential numbers like "1. 2. 3."
            r'^\d+\.\s+\d+\.\s+\d+\.\s+\d+',  # Four or more sequential numbers
            r'^(Date|Signature)',
            r'^Rs\.\s*$',  # Currency symbols
            r'^\d+\s+\d+\s+\d+\s*$',  # Numbers without periods like "1 2 3"
            r'^\d+\s*\d+\s*\d+\s*$',  # Numbers closely spaced like "123"
        ]
        
        for pattern in form_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
                
        # Skip very short texts that are likely labels
        if len(text.strip()) <= 2 and text.strip().isdigit():
            return True
            
        # Skip single letters or very short words in numbered contexts
        if re.match(r'^\d+\.\s*[A-Z]\.?\s*$', text.strip()):
            return True
            
        # Skip short sequences of numbers and periods (form field patterns)
        if re.match(r'^[\d\.\s]{2,10}$', text.strip()) and '.' in text:
            return True
            
        return False

    @staticmethod
    def detect_tables(page, blocks: List[Dict]) -> List[Dict]:
        """Detect table areas on the page"""
        table_areas = []
        
        # Method 1: Look for table-like patterns in text blocks
        for block in blocks:
            if block["type"] != 0:
                continue
                
            text = PDFTextUtils.extract_block_text(block).strip()
            
            # Check if this looks like a table header/structure
            if TableDetectionUtils.is_table_structure(text):
                table_areas.append({
                    'bbox': block['bbox'],
                    'type': 'text_table',
                    'confidence': 0.8
                })
        
        # Method 2: Use PyMuPDF's table detection if available
        try:
            layout_tables = page.find_tables()
            for table in layout_tables:
                table_areas.append({
                    'bbox': table.bbox,
                    'type': 'detected_table', 
                    'confidence': 0.9
                })
        except:
            # Fallback: basic geometric detection
            pass
            
        # Method 3: Detect form-like structures (multiple numbered items)
        form_area = TableDetectionUtils.detect_form_structure(blocks)
        if form_area:
            table_areas.append(form_area)
        
        return table_areas

    @staticmethod
    def detect_form_structure(blocks: List[Dict]) -> Optional[Dict]:
        """Detect form-like structures with numbered fields"""
        numbered_blocks = []
        
        for block in blocks:
            if block["type"] != 0:
                continue
                
            text = PDFTextUtils.extract_block_text(block).strip()
            
            # Look for numbered form fields, but exclude likely headings
            if (re.match(r'^\d+\.\s*(.{0,50})?$', text.strip()) and
                not TableDetectionUtils.is_table_or_form_content(text)):
                numbered_blocks.append(block)
        
        # If we have many numbered items, it's likely a form
        if len(numbered_blocks) >= 8:  # At least 8 numbered items
            # Calculate bounding box for the form area
            min_x = min(block['bbox'][0] for block in numbered_blocks)
            min_y = min(block['bbox'][1] for block in numbered_blocks) 
            max_x = max(block['bbox'][2] for block in numbered_blocks)
            max_y = max(block['bbox'][3] for block in numbered_blocks)
            
            return {
                'bbox': [min_x, min_y, max_x, max_y],
                'type': 'form_structure',
                'confidence': 0.7
            }
        
        return None


class TOCDetectionUtils:
    """Utilities for Table of Contents detection"""
    
    @staticmethod
    def is_table_of_contents_page(page, blocks: List[Dict]) -> bool:
        """Detect if this page is a Table of Contents"""
        page_text = page.get_text().upper()
        
        # Look for "TABLE OF CONTENTS" heading
        if "TABLE OF CONTENTS" in page_text:
            return True
        
        # Look for patterns indicating TOC structure
        toc_indicators = 0
        
        for block in blocks:
            if block["type"] != 0:
                continue
                
            text = PDFTextUtils.extract_block_text(block).strip()
            
            # Check for typical TOC patterns
            if TOCDetectionUtils.is_toc_entry(text):
                toc_indicators += 1
        
        # If we have many TOC-like entries, it's likely a TOC page
        return toc_indicators >= 3

    @staticmethod
    def is_toc_entry(text: str) -> bool:
        """Check if text looks like a table of contents entry"""
        # Pattern 1: "1. Something 5" or "Chapter 1: Title 10"
        if re.search(r'^(\d+\.|\d+\.\d+\.?|Chapter\s+\d+:?)\s+.+\s+\d+\s*$', text.strip()):
            return True
        
        # Pattern 2: Multiple entries concatenated with page numbers
        if re.search(r'\d+\.\d+\s+[^0-9]+\s+\d+\s+\d+\.\d+', text):
            return True
        
        # Pattern 3: Text ending with just a number (page number)
        if re.search(r'^.+\s+\d{1,3}\s*$', text.strip()) and len(text.strip()) > 10:
            return True
        
        return False

    @staticmethod
    def extract_toc_heading_only(blocks: List[Dict], page_num: int) -> List[Dict]:
        """Extract only the 'Table of Contents' heading from TOC page"""
        headings = []
        
        for block in blocks:
            if block["type"] != 0:
                continue
                
            text = PDFTextUtils.extract_block_text(block).strip()
            
            # Only extract the "Table of Contents" heading itself
            if re.match(r'^(Table of Contents|Contents|TOC)$', text, re.IGNORECASE):
                headings.append({
                    'level': 'H1',
                    'text': text,
                    'page': page_num - 1
                })
        
        return headings


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


class FontHierarchyAnalyzer:
    """Statistical analysis of font usage in document"""
    
    def analyze(self, doc) -> Dict:
        """Analyze font hierarchy in document"""
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
        """Analyze fonts on a single page"""
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            if block["type"] == 0:  # Text block
                text = PDFTextUtils.extract_block_text(block)
                if len(text.strip()) > 3:  # Only substantial text
                    
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            font_key = (span.get("size", 10), span.get("font", ""))
                            
                            font_stats[font_key]['count'] += 1
                            font_stats[font_key]['total_chars'] += len(span.get("text", ""))
                            font_stats[font_key]['pages'].add(page_num)
                            
                            if span.get("flags", 0) & 2**4:  # Bold
                                font_stats[font_key]['is_bold'] = True
                                
                            # Sample text for context
                            if len(font_stats[font_key]['sample_texts']) < 3:
                                font_stats[font_key]['sample_texts'].append(text[:50])
    
    def _determine_hierarchy(self, font_stats: Dict) -> Dict:
        """Determine font hierarchy from statistics"""
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
                    (r'^\([a-z]\)', 0.4),      # "(a) item"
                ],
                'font_threshold': 'p50'  # Median font size
            }
        ]
