# main.py
import os
import json
import time
from pathlib import Path
from typing import Dict

from pdf_analyzer import PDFAnalyzer
from rule_engine import SmartRuleEngine
from ml_engine import MLEngine
from utils import setup_logging, measure_time

class PDFOutlineExtractor:
    """Main orchestrator for PDF outline extraction"""
    
    def __init__(self):
        self.logger = setup_logging()
        self.analyzer = PDFAnalyzer()
        self.rule_engine = SmartRuleEngine()
        self.ml_engine = None  # Lazy load
    
    @measure_time
    def process_pdf(self, pdf_path: str) -> Dict:
        """Process single PDF file"""
        self.logger.info(f"Processing: {pdf_path}")
        
        try:
            # Step 1: Analyze PDF type (< 0.5s)
            analysis = self.analyzer.analyze(pdf_path)
            self.logger.info(f"PDF Analysis: {analysis['is_standard']} (standard)")
            
            # Step 2: Choose processing method
            if analysis['is_standard']:
                result = self.rule_engine.extract(pdf_path)
                self.logger.info("Processed with rule engine")
            else:
                # Lazy load ML engine
                if self.ml_engine is None:
                    self.logger.info("Loading ML engine...")
                    self.ml_engine = MLEngine()
                
                result = self.ml_engine.extract(pdf_path)
                self.logger.info("Processed with ML engine")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing {pdf_path}: {str(e)}")
            # Return empty structure on error
            return {
                "title": "Error Processing Document",
                "outline": []
            }
    
    def process_directory(self, input_dir: str, output_dir: str):
        """Process all PDFs in directory"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        pdf_files = list(input_path.glob("*.pdf"))
        self.logger.info(f"Found {len(pdf_files)} PDF files")
        
        for pdf_file in pdf_files:
            start_time = time.time()
            
            # Process PDF
            result = self.process_pdf(str(pdf_file))
            
            # Save result
            output_file = output_path / f"{pdf_file.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            processing_time = time.time() - start_time
            self.logger.info(f"Completed {pdf_file.name} in {processing_time:.2f}s")

def main():
    """Main entry point for Docker container"""
    extractor = PDFOutlineExtractor()
    
    # Process all PDFs from input to output directory
    extractor.process_directory('/app/input', '/app/output')

if __name__ == "__main__":
    main()