import fitz
import joblib
from typing import Dict
from .ocr_handler import OCRHandler
from .pdf_scanner import PDFScanner
from .block_extractor import BlockExtractor
from .ml_classifier import MLClassifier
from .heuristic_classifier import HeuristicClassifier
from .fallback_strategies import FallbackStrategies
from .output_formatter import OutputFormatter

class MLEngine:
    """Handles irregular PDFs with ML/OCR approaches"""
    
    def __init__(self):
        self.ocr_engine = None
        self.layout_model = None
        self._lazy_load_models()
        
        # Initialize components
        self.ocr_handler = OCRHandler()
        self.pdf_scanner = PDFScanner()
        self.block_extractor = BlockExtractor()
        self.ml_classifier = MLClassifier()
        self.heuristic_classifier = HeuristicClassifier()
        self.fallback_strategies = FallbackStrategies()
        self.output_formatter = OutputFormatter()
    
    def _lazy_load_models(self):
        """Load models only when needed"""
        try:
            # Try to load pre-trained models
            self.feature_extractor = joblib.load('models/feature_extractor.pkl')
            self.classifier = joblib.load('models/heading_classifier.pkl')
            print("✅ ML Engine: Successfully loaded pre-trained ML models")
        except Exception as e:
            # Fallback to simple implementation
            print(f"⚠️  ML Engine: Failed to load ML models ({e}), will use heuristic fallback")
            self.feature_extractor = None
            self.classifier = None
    
    def extract(self, pdf_path: str) -> Dict:
        """Extract headings from irregular PDFs"""
        doc = fitz.open(pdf_path)
        
        # Determine processing strategy
        if self.pdf_scanner.is_scanned_pdf(doc):
            return self._process_scanned_pdf(pdf_path, doc)
        else:
            return self._process_complex_layout(doc)
    
    def _process_scanned_pdf(self, pdf_path: str, doc) -> Dict:
        """Process scanned PDFs using OCR or enhanced extraction"""
        # Try OCR first
        if self.ocr_handler.setup_ocr_if_needed():
            return self.ocr_handler.process_scanned_pdf_with_ocr(pdf_path, doc)
        else:
            # Fallback to enhanced text extraction
            return self.fallback_strategies.process_scanned_pdf_fallback(doc)
    
    def _process_complex_layout(self, doc) -> Dict:
        """Handle PDFs with complex layouts using ML features"""
        all_blocks = []
        
        # Extract all text blocks with features
        for page_num, page in enumerate(doc):
            blocks = self.block_extractor.extract_blocks_with_features(page, page_num)
            all_blocks.extend(blocks)
        
        if not all_blocks:
            # If no blocks extracted, try fallback
            print("⚠️  ML Engine: No text blocks extracted, using fallback strategies")
            return self.fallback_strategies.handle_corrupted_pdf_fallback(doc)
        
        # Use classifier if available
        if self.classifier and self.feature_extractor:
            try:
                ml_predictions = self.ml_classifier.classify_blocks_with_ml(
                    all_blocks, self.classifier, self.feature_extractor
                )
                heuristic_predictions = self.heuristic_classifier.classify(all_blocks)
                
                # Use hybrid approach: combine ML and heuristics
                predictions = self.ml_classifier.combine_predictions(
                    ml_predictions, heuristic_predictions
                )
                print(f"✅ ML Engine: Using hybrid ML+heuristic approach: {len(predictions)} total headings")
                
            except Exception as e:
                print(f"❌ ML Engine: ML classification failed: {e}, falling back to heuristics")
                predictions = self.heuristic_classifier.classify(all_blocks)
        else:
            # Fallback to advanced heuristics
            print("⚠️  ML Engine: ML models not available, using advanced heuristics")
            predictions = self.heuristic_classifier.classify(all_blocks)
        
        # Convert to output format
        return self.output_formatter.format_output(predictions, all_blocks)