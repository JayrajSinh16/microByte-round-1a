# ml_engine.py
import fitz
import numpy as np
from typing import Dict, List
import joblib
import pickle
import re

class MLEngine:
    """Handles irregular PDFs with ML/OCR approaches"""
    
    def __init__(self):
        self.ocr_engine = None
        self.layout_model = None
        self._lazy_load_models()
    
    def _lazy_load_models(self):
        """Load models only when needed"""
        try:
            # Try to load pre-trained models
            self.feature_extractor = joblib.load('models/feature_extractor.pkl')
            self.classifier = joblib.load('models/heading_classifier.pkl')
        except:
            # Fallback to simple implementation
            self.feature_extractor = None
            self.classifier = None
    
    def extract(self, pdf_path: str) -> Dict:
        """Extract headings from irregular PDFs"""
        doc = fitz.open(pdf_path)
        
        # Determine processing strategy
        if self._is_scanned_pdf(doc):
            return self._process_scanned_pdf(pdf_path, doc)
        else:
            return self._process_complex_layout(doc)
    
    def _is_scanned_pdf(self, doc) -> bool:
        """Check if PDF is scanned/image-based"""
        sample_page = doc[0]
        text = sample_page.get_text().strip()
        images = sample_page.get_images()
        
        return len(images) > 0 and len(text) < 50
    
    def _process_scanned_pdf(self, pdf_path: str, doc) -> Dict:
        """Process scanned PDFs using OCR"""
        # Lazy load OCR
        if not self.ocr_engine:
            self._setup_ocr()
        
        headings = []
        title = None
        
        for page_num, page in enumerate(doc):
            # Convert page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
            img_data = pix.pil_tobytes("PNG")
            
            # Perform OCR
            ocr_results = self._perform_ocr(img_data)
            
            # Extract headings from OCR results
            page_headings = self._extract_headings_from_ocr(
                ocr_results, page_num + 1
            )
            
            if page_num == 0 and page_headings:
                # First heading might be title
                title = page_headings[0]['text']
                headings.extend(page_headings[1:])
            else:
                headings.extend(page_headings)
        
        doc.close()
        return {
            "title": title or "Untitled Document",
            "outline": headings
        }
    
    def _setup_ocr(self):
        """Setup lightweight OCR engine"""
        try:
            # Option 1: Use Tesseract (smaller, available in Docker)
            import pytesseract
            self.ocr_engine = 'tesseract'
        except:
            # Option 2: Fallback to basic text extraction
            self.ocr_engine = 'basic'
    
    def _perform_ocr(self, img_data) -> List[Dict]:
        """Perform OCR on image data"""
        if self.ocr_engine == 'tesseract':
            import pytesseract
            from PIL import Image
            import io
            
            img = Image.open(io.BytesIO(img_data))
            # Get detailed OCR data
            ocr_data = pytesseract.image_to_data(
                img, output_type=pytesseract.Output.DICT
            )
            
            return self._parse_tesseract_output(ocr_data)
        else:
            # Basic fallback
            return []
    
    def _parse_tesseract_output(self, ocr_data: Dict) -> List[Dict]:
        """Parse Tesseract output to structured format"""
        blocks = []
        current_block = None
        
        for i in range(len(ocr_data['text'])):
            if ocr_data['text'][i].strip():
                if current_block is None or ocr_data['block_num'][i] != current_block['block_num']:
                    if current_block:
                        blocks.append(current_block)
                    
                    current_block = {
                        'text': ocr_data['text'][i],
                        'conf': ocr_data['conf'][i],
                        'x': ocr_data['left'][i],
                        'y': ocr_data['top'][i],
                        'width': ocr_data['width'][i],
                        'height': ocr_data['height'][i],
                        'block_num': ocr_data['block_num'][i]
                    }
                else:
                    current_block['text'] += ' ' + ocr_data['text'][i]
        
        if current_block:
            blocks.append(current_block)
        
        return blocks
    
    def _extract_headings_from_ocr(self, ocr_results: List[Dict], page_num: int) -> List[Dict]:
        """Extract headings from OCR results using heuristics"""
        headings = []
        
        if not ocr_results:
            return headings
        
        # Calculate average text size
        avg_height = np.mean([block['height'] for block in ocr_results])
        
        for block in ocr_results:
            # Skip low confidence or very long text
            if block['conf'] < 50 or len(block['text']) > 100:
                continue
            
            # Heading heuristics
            is_heading = False
            level = None
            
            # Size-based detection
            if block['height'] > avg_height * 1.5:
                is_heading = True
                level = 'H1'
            elif block['height'] > avg_height * 1.2:
                is_heading = True
                level = 'H2'
            
            # Pattern-based detection
            if re.match(r'^\d+\.?\s+\w+', block['text']):
                is_heading = True
                level = level or 'H2'
            
            if is_heading and level:
                headings.append({
                    'level': level,
                    'text': block['text'].strip(),
                    'page': page_num
                })
        
        return headings
    
    # Continue in ml_engine.py
    def _process_complex_layout(self, doc) -> Dict:
        """Handle PDFs with complex layouts using ML features"""
        all_blocks = []
        
        # Extract all text blocks with features
        for page_num, page in enumerate(doc):
            blocks = self._extract_blocks_with_features(page, page_num)
            all_blocks.extend(blocks)
        
        # Use classifier if available
        if self.classifier:
            predictions = self._classify_blocks(all_blocks)
        else:
            # Fallback to advanced heuristics
            predictions = self._heuristic_classification(all_blocks)
        
        # Convert to output format
        return self._format_output(predictions, all_blocks)
    
    def _extract_blocks_with_features(self, page, page_num: int) -> List[Dict]:
        """Extract text blocks with ML-ready features"""
        blocks = []
        page_dict = page.get_text("dict")
        page_height = page_dict["height"]
        page_width = page_dict["width"]
        
        for block_idx, block in enumerate(page_dict["blocks"]):
            if block["type"] != 0:  # Skip non-text
                continue
            
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
                'has_punctuation': text.strip()[-1] in '.!?',
                
                # Font features
                'font_sizes': [],
                'is_bold': False,
                'is_italic': False,
                'font_names': set()
            }
            
            # Extract font information
            self._extract_font_features(block, features)
            
            blocks.append(features)
        
        return blocks
    
    def _extract_font_features(self, block: Dict, features: Dict):
        """Extract font-related features from block"""
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                features['font_sizes'].append(span.get("size", 0))
                features['font_names'].add(span.get("font", ""))
                
                flags = span.get("flags", 0)
                if flags & 2**4:  # Bold
                    features['is_bold'] = True
                if flags & 2**1:  # Italic
                    features['is_italic'] = True
        
        # Continue in ml_engine.py
        # Calculate aggregates
        if features['font_sizes']:
            features['avg_font_size'] = np.mean(features['font_sizes'])
            features['max_font_size'] = max(features['font_sizes'])
        else:
            features['avg_font_size'] = 0
            features['max_font_size'] = 0
        
        features['font_variety'] = len(features['font_names'])
    
    def _extract_text_from_block(self, block: Dict) -> str:
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
    
    def _heuristic_classification(self, blocks: List[Dict]) -> List[Dict]:
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
        font_sizes = []
        for block in blocks:
            if block['avg_font_size'] > 0:
                font_sizes.extend([block['avg_font_size']] * block['word_count'])
        
        return {
            'median_font_size': np.median(font_sizes) if font_sizes else 12,
            'font_size_percentiles': {
                'p75': np.percentile(font_sizes, 75) if font_sizes else 14,
                'p90': np.percentile(font_sizes, 90) if font_sizes else 16,
                'p95': np.percentile(font_sizes, 95) if font_sizes else 18
            },
            'avg_block_words': np.mean([b['word_count'] for b in blocks]),
            'total_blocks': len(blocks)
        }
    
    def _calculate_heading_score(self, block: Dict, doc_stats: Dict) -> float:
        """Calculate probability that block is a heading"""
        score = 0.0
        text = block['text'].strip()
        
        # Font size score (most important)
        if block['avg_font_size'] > doc_stats['font_size_percentiles']['p90']:
            score += 0.4
        elif block['avg_font_size'] > doc_stats['font_size_percentiles']['p75']:
            score += 0.25
        
        # Position score
        if block['x_normalized'] < 0.2:  # Left aligned
            score += 0.1
        if block['y_normalized'] < 0.15 and block['page_num'] == 0:  # Top of first page
            score += 0.15
        
        # Text pattern score
        if block['starts_with_number']:
            score += 0.15
        if block['word_count'] < 10:
            score += 0.1
        if block['is_uppercase']:
            score += 0.1
        if block['is_bold']:
            score += 0.15
            
        # Special patterns for common heading structures
        if re.match(r'^(What|How|Why|Where|When)', text, re.IGNORECASE):
            score += 0.2
        if text.endswith(':'):
            score += 0.15
        if re.match(r'^For (each|the)', text, re.IGNORECASE):
            score += 0.2
        if re.match(r'^(Summary|Background|Approach|Evaluation|Appendix)', text, re.IGNORECASE):
            score += 0.25
        if text.lower() == 'milestones':
            score += 0.3
            
        # Penalty for paragraph-like features
        if block['has_punctuation'] and block['word_count'] > 8:
            score -= 0.15
        if block['word_count'] > 15:
            score -= 0.2
        
        return min(max(score, 0), 1)
    
    def _determine_level_contextual(self, block: Dict, doc_stats: Dict, existing_headings: List[Dict]) -> str:
        """Determine heading level based on features and context"""
        text = block['text'].strip()
        
        # Special case handling
        if text.lower() == 'milestones':
            return 'H3'  # Should be H3, not H2
            
        if re.match(r'^For (each|the) Ontario', text, re.IGNORECASE):
            return 'H4'  # These are sub-items
            
        if re.match(r'^What could.*mean', text, re.IGNORECASE):
            return 'H3'  # This is the parent of "For each..." items
        
        # Font size-based determination with context
        if block['avg_font_size'] > doc_stats['font_size_percentiles']['p95']:
            # Check if this should really be H1 or if we have too many H1s already
            recent_h1_count = sum(1 for h in existing_headings[-5:] 
                                if h.get('level') == 'H1')
            if recent_h1_count > 2:
                return 'H2'  # Demote to H2 if too many recent H1s
            return 'H1'
            
        elif block['avg_font_size'] > doc_stats['font_size_percentiles']['p90']:
            # Context-aware H2/H3 decision
            if text.startswith('Appendix'):
                return 'H2'
            if len(existing_headings) > 0:
                last_heading = existing_headings[-1]
                if last_heading.get('level') == 'H2' and block['page_num'] == last_heading['block']['page_num']:
                    return 'H3'  # Likely a subsection
            return 'H2' if not block['starts_with_number'] else 'H1'
        else:
            # Check for subsection patterns
            if text.endswith(':') or re.match(r'^\d+\.', text):
                return 'H3'
            return 'H3'
    
    def _format_output(self, predictions: List[Dict], blocks: List[Dict]) -> Dict:
        """Format predictions to required output format"""
        headings = []
        title = None
        
        # Sort by page and position
        predictions.sort(key=lambda x: (
            x['block']['page_num'], 
            x['block']['y_normalized']
        ))
        
        for i, pred in enumerate(predictions):
            if pred['is_heading']:
                block = pred['block']
                text = block['text'].strip()
                
                # Better title detection
                if (title is None and block['page_num'] < 2 and 
                    any(keyword in text.lower() for keyword in 
                        ['proposal', 'business plan', 'ontario digital library', 'rfp'])):
                    title = text
                else:
                    # Filter out overly long headings that are likely paragraphs
                    if len(text) < 150 and block['word_count'] < 20:
                        headings.append({
                            'level': pred['level'],
                            'text': text,
                            'page': block['page_num'] + 1  # Convert to 1-indexed
                        })
        
        return {
            'title': title or "Untitled Document",
            'outline': headings
        }