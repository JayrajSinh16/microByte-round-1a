# tests/benchmark.py
import time
import psutil
import os
from pathlib import Path
import pandas as pd
from main import PDFOutlineExtractor

class PerformanceBenchmark:
    def __init__(self):
        self.extractor = PDFOutlineExtractor()
        self.results = []
    
    def benchmark_pdf(self, pdf_path: str, pdf_type: str):
        """Benchmark single PDF processing"""
        # Get file size
        file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
        
        # Memory before
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)
        
        # Time processing
        start_time = time.time()
        result = self.extractor.process_pdf(pdf_path)
        processing_time = time.time() - start_time
        
        # Memory after
        mem_after = process.memory_info().rss / (1024 * 1024)
        mem_used = mem_after - mem_before
        
        # Record results
        self.results.append({
            'pdf_name': Path(pdf_path).name,
            'pdf_type': pdf_type,
            'file_size_mb': round(file_size_mb, 2),
            'processing_time': round(processing_time, 3),
            'memory_used_mb': round(mem_used, 2),
            'headings_found': len(result['outline']),
            'pages': self._get_page_count(pdf_path)
        })
    
    def _get_page_count(self, pdf_path: str) -> int:
        import fitz
        doc = fitz.open(pdf_path)
        count = len(doc)
        doc.close()
        return count
    
    def run_benchmarks(self):
        """Run benchmarks on various PDF types"""
        test_pdfs = [
            ("test_data/1_page.pdf", "single_page"),
            ("test_data/10_pages.pdf", "small"),
            ("test_data/25_pages.pdf", "medium"),
            ("test_data/50_pages.pdf", "large"),
            ("test_data/scanned_10pages.pdf", "scanned"),
            ("test_data/complex_layout.pdf", "complex"),
            ("test_data/japanese.pdf", "multilingual")
        ]
        
        for pdf_path, pdf_type in test_pdfs:
            if Path(pdf_path).exists():
                print(f"Benchmarking {pdf_path}...")
                self.benchmark_pdf(pdf_path, pdf_type)
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate performance report"""
        df = pd.DataFrame(self.results)
        
        print("\n=== Performance Benchmark Results ===")
        print(df.to_string(index=False))
        
        print("\n=== Summary Statistics ===")
        print(f"Average processing time: {df['processing_time'].mean():.2f}s")
        print(f"Max processing time: {df['processing_time'].max():.2f}s")
        print(f"Average memory usage: {df['memory_used_mb'].mean():.2f}MB")
        
        # Check if all PDFs processed under 10 seconds
        if df['processing_time'].max() < 10:
            print("\n✅ All PDFs processed within 10-second limit!")
        else:
            print("\n❌ Some PDFs exceeded 10-second limit!")
            print(df[df['processing_time'] >= 10][['pdf_name', 'processing_time']])

if __name__ == "__main__":
    benchmark = PerformanceBenchmark()
    benchmark.run_benchmarks()