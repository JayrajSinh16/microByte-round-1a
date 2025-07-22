import fitz
import re
import numpy as np
from typing import Dict, List
from shared_utils import TextNormalizationUtils

class OCRHandler:
    """Handles OCR operations for scanned PDFs"""
    
    def __init__(self):
        self.ocr_engine = None
    
    def setup_ocr_if_needed(self) -> bool:
        """Setup OCR only when needed and return success status"""
        if not self.ocr_engine:
            self._setup_ocr()
        return self.ocr_engine == 'tesseract'
    
    def _setup_ocr(self):
        """Setup lightweight OCR engine"""
        try:
            # Option 1: Use Tesseract (smaller, available in Docker)
            import pytesseract
            import io
            from PIL import Image
            # Test if tesseract is available
            pytesseract.get_tesseract_version()
            self.ocr_engine = 'tesseract'
        except:
            # Option 2: Fallback to basic text extraction
            self.ocr_engine = 'basic'
    
    def process_scanned_pdf_with_ocr(self, pdf_path: str, doc) -> Dict:
        """Process scanned PDFs using OCR"""
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
        
        return {
            "title": title or "Untitled Document",
            "outline": headings
        }
    
    def _perform_ocr(self, img_data) -> List[Dict]:
        """Perform OCR on image data"""
        if self.ocr_engine == 'tesseract':
            try:
                import pytesseract
                from PIL import Image
                import io
                
                img = Image.open(io.BytesIO(img_data))
                # Get detailed OCR data
                ocr_data = pytesseract.image_to_data(
                    img, output_type=pytesseract.Output.DICT
                )
                
                return self._parse_tesseract_output(ocr_data)
            except Exception as e:
                print(f"OCR failed: {e}, falling back to basic extraction")
                self.ocr_engine = 'basic'
                return []
        else:
            # Basic fallback - enhanced text extraction
            return self._basic_text_extraction(img_data)
    
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
    
    def _basic_text_extraction(self, img_data) -> List[Dict]:
        """Basic text extraction fallback when OCR is not available"""
        # This is a placeholder - in practice, we'd fall back to 
        # enhanced PyMuPDF text extraction with layout analysis
        return []
    
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