from typing import Dict


class PDFScanner:
    """Handles PDF scanning and analysis"""
    
    def is_scanned_pdf(self, doc) -> bool:
        """Check if PDF is scanned/image-based"""
        sample_page = doc[0]
        text = sample_page.get_text().strip()
        images = sample_page.get_images()
        
        return len(images) > 0 and len(text) < 50
    
    def analyze_pdf_structure(self, doc) -> Dict:
        """Analyze PDF structure to infer content type"""
        analysis = {'likely_stem_document': False}
        
        # Check PDF metadata
        metadata = doc.metadata
        title = metadata.get('title', '').lower()
        
        # Look for STEM-related indicators in title
        if any(keyword in title for keyword in ['stem', 'pathway', 'parsippany', 'troy', 'hills']):
            analysis['likely_stem_document'] = True
        
        # Check page count and dimensions
        analysis['page_count'] = len(doc)
        if len(doc) > 0:
            page = doc[0]
            analysis['page_width'] = page.bound().width
            analysis['page_height'] = page.bound().height
            
        return analysis