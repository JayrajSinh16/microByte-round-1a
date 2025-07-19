# pdf_analyzer.py
import fitz
import time
from typing import Dict, Any
from config import Config

class PDFAnalyzer:
    """Quick PDF type detection in <0.5 seconds"""
    
    def __init__(self):
        self.analysis_cache = {}
    
    def analyze(self, pdf_path: str) -> Dict[str, Any]:
        start_time = time.time()
        
        # Check cache first
        if pdf_path in self.analysis_cache:
            return self.analysis_cache[pdf_path]
        
        doc = fitz.open(pdf_path)
        metrics = self._initialize_metrics()
        
        # Smart sampling: first, middle, last pages
        sample_pages = self._get_sample_pages(len(doc))
        
        for page_idx in sample_pages:
            self._analyze_page(doc[page_idx], metrics)
        
        # Calculate final metrics
        metrics['is_standard'] = self._is_standard_pdf(metrics)
        metrics['analysis_time'] = time.time() - start_time
        
        # Cache results
        self.analysis_cache[pdf_path] = metrics
        doc.close()
        
        return metrics
    
    def _initialize_metrics(self) -> Dict[str, Any]:
        return {
            'total_pages': 0,
            'has_scanned_pages': False,
            'text_extraction_rate': 1.0,
            'font_variety': set(),
            'avg_text_blocks_per_page': 0,
            'has_toc': False,
            'has_images': False,
            'layout_complexity': 0.0,
            'font_sizes': [],
            'is_standard': True
        }
    
    def _get_sample_pages(self, total_pages: int) -> list:
        if total_pages <= 3:
            return list(range(total_pages))
        elif total_pages <= 10:
            return [0, total_pages//2, total_pages-1]
        else:
            # Sample more pages for longer documents
            return [0, 2, total_pages//4, total_pages//2, total_pages-2]
    
    def _analyze_page(self, page, metrics: Dict):
        # Get page content
        text = page.get_text()
        blocks = page.get_text("dict")["blocks"]
        images = page.get_images()
        
        # Check for scanned content
        if images and len(text.strip()) < 50:
            metrics['has_scanned_pages'] = True
        
        # Analyze text blocks
        text_blocks = 0
        for block in blocks:
            if block["type"] == 0:  # Text block
                text_blocks += 1
                self._analyze_block(block, metrics)
        
        metrics['avg_text_blocks_per_page'] += text_blocks
        
    def _analyze_block(self, block: Dict, metrics: Dict):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                metrics['font_variety'].add(span.get("font"))
                metrics['font_sizes'].append(span.get("size", 0))
    
    def _is_standard_pdf(self, metrics: Dict) -> bool:
        """Determine if PDF needs ML processing"""
        
        # Convert set to count
        font_count = len(metrics['font_variety'])
        
        # Decision logic
        if metrics['has_scanned_pages']:
            return False
        if font_count > Config.MAX_FONT_VARIETY:
            return False
        if metrics['text_extraction_rate'] < Config.MIN_TEXT_EXTRACTION_RATE:
            return False
            
        return True