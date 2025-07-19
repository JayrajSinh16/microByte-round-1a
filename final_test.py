# final_test.py - Comprehensive final testing
import os
import json
import time
import sys
from pathlib import Path

# Add src to path to import modules
sys.path.append('src')
from main import PDFOutlineExtractor

class FinalTester:
    """Final validation before submission"""
    
    def __init__(self):
        self.extractor = PDFOutlineExtractor()
        self.test_results = []
    
    def run_all_tests(self):
        """Run all validation tests"""
        print("🚀 Running Final Validation Tests...")
        
        # Test 1: Basic functionality
        self.test_basic_functionality()
        
        # Test 2: Performance compliance
        self.test_performance_compliance()
        
        # Test 3: Edge cases
        self.test_edge_cases()
        
        # Test 4: Output format validation
        self.test_output_format()
        
        # Test 5: Docker compliance
        self.test_docker_compliance()
        
        # Generate report
        self.generate_final_report()
    
    def test_basic_functionality(self):
        """Test basic extraction functionality"""
        print("\n📋 Test 1: Basic Functionality")
        
        test_files = [
            "sample.pdf",
            "academic_paper.pdf",
            "technical_manual.pdf"
        ]
        
        for pdf_file in test_files:
            pdf_path = f"test_data/{pdf_file}"
            if Path(pdf_path).exists():
                result = self.extractor.process_pdf(pdf_path)
                
                test_result = {
                    'test': 'basic_functionality',
                    'file': pdf_file,
                    'passed': bool(result['title'] and result['outline']),
                    'heading_count': len(result['outline'])
                }
                
                self.test_results.append(test_result)
                print(f"  ✓ {pdf_file}: {test_result['heading_count']} headings found")
    
    def test_performance_compliance(self):
        """Test performance requirements"""
        print("\n⚡ Test 2: Performance Compliance")
        
        # Test 50-page document
        large_pdf = "test_data/50_page_document.pdf"
        if Path(large_pdf).exists():
            start_time = time.time()
            result = self.extractor.process_pdf(large_pdf)
            processing_time = time.time() - start_time
            
            passed = processing_time < 10
            
            test_result = {
                'test': 'performance',
                'file': '50_page_document.pdf',
                'passed': passed,
                'processing_time': round(processing_time, 2),
                'limit': 10
            }
            
            self.test_results.append(test_result)
            
            status = "✓" if passed else "✗"
            print(f"  {status} 50-page PDF: {processing_time:.2f}s (limit: 10s)")
    
    def test_edge_cases(self):
        """Test edge cases"""
        print("\n🔧 Test 3: Edge Cases")
        
        edge_cases = [
            ("empty.pdf", "Empty PDF"),
            ("scanned.pdf", "Scanned PDF"),
            ("rotated.pdf", "Rotated text"),
            ("multi_column.pdf", "Multi-column layout"),
            ("japanese.pdf", "Non-English text")
        ]
        
        for pdf_file, description in edge_cases:
            pdf_path = f"test_data/{pdf_file}"
            if Path(pdf_path).exists():
                try:
                    result = self.extractor.process_pdf(pdf_path)
                    passed = isinstance(result, dict) and 'title' in result
                    error = None
                except Exception as e:
                    passed = False
                    error = str(e)
                
                test_result = {
                    'test': 'edge_case',
                    'file': pdf_file,
                    'description': description,
                    'passed': passed,
                    'error': error
                }
                
                self.test_results.append(test_result)
                
                status = "✓" if passed else "✗"
                print(f"  {status} {description}: {'Passed' if passed else error}")
    
    def test_output_format(self):
        """Validate output format"""
        print("\n📐 Test 4: Output Format Validation")
        
        pdf_path = "test_data/sample.pdf"
        if Path(pdf_path).exists():
            result = self.extractor.process_pdf(pdf_path)
            
            # Validate structure
            format_checks = {
                'has_title': 'title' in result,
                'has_outline': 'outline' in result,
                'outline_is_list': isinstance(result.get('outline', None), list),
                'valid_heading_format': all(
                    'level' in h and 'text' in h and 'page' in h
                    for h in result.get('outline', [])
                ),
                'valid_levels': all(
                    h['level'] in ['H1', 'H2', 'H3']
                    for h in result.get('outline', [])
                ),
                'valid_json': self._validate_json_serializable(result)
            }
            
            all_passed = all(format_checks.values())
            
            test_result = {
                'test': 'output_format',
                'passed': all_passed,
                'checks': format_checks
            }
            
            self.test_results.append(test_result)
            
            for check, passed in format_checks.items():
                status = "✓" if passed else "✗"
                print(f"  {status} {check}")
    
    def _validate_json_serializable(self, obj):
        """Check if object is JSON serializable"""
        try:
            json.dumps(obj, ensure_ascii=False)
            return True
        except:
            return False
    
    def test_docker_compliance(self):
        """Test Docker requirements"""
        print("\n🐳 Test 5: Docker Compliance")
        
        checks = {
            'dockerfile_exists': Path('Dockerfile').exists(),
            'requirements_exists': Path('requirements.txt').exists(),
            'amd64_compatible': self._check_dockerfile_arch(),
            'no_gpu_deps': self._check_no_gpu(),
            'size_compliant': self._estimate_size() < 500  # MB
        }
        
        for check, passed in checks.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check}")
        
        test_result = {
            'test': 'docker_compliance',
            'passed': all(checks.values()),
            'checks': checks
        }
        
        self.test_results.append(test_result)
    
    def _check_dockerfile_arch(self):
        """Check if Dockerfile is AMD64 compatible"""
        if Path('Dockerfile').exists():
            with open('Dockerfile', 'r') as f:
                content = f.read()
                # Check for ARM-specific images
                return 'arm' not in content.lower()
        return False
    
    def _check_no_gpu(self):
        """Check for GPU dependencies"""
        if Path('requirements.txt').exists():
            with open('requirements.txt', 'r') as f:
                content = f.read().lower()
                gpu_packages = ['cuda', 'cudnn', 'tensorflow-gpu', 'torch-gpu']
                return not any(pkg in content for pkg in gpu_packages)
        return True
    
    def _estimate_size(self):
        """Estimate Docker image size"""
        size_mb = 0
        
        # Base image size
        size_mb += 150  # python:3.9-slim
        
        # Dependencies from requirements.txt
        if Path('requirements.txt').exists():
            with open('requirements.txt', 'r') as f:
                deps = f.readlines()
                # Rough estimates
                for dep in deps:
                    if 'pymupdf' in dep.lower():
                        size_mb += 20
                    elif 'numpy' in dep.lower():
                        size_mb += 15
                    elif 'scikit-learn' in dep.lower():
                        size_mb += 60
                    else:
                        size_mb += 5
        
        # Application code
        size_mb += 10
        
        return size_mb
    
    def generate_final_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*50)
        print("📊 FINAL TEST REPORT")
        print("="*50)
        
        # Calculate summary
        total_tests = len(self.test_results)
        passed_tests = sum(1 for t in self.test_results if t['passed'])
        
        print(f"\nTotal Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        # Detailed results by category
        print("\n📋 Detailed Results:")
        
        categories = {}
        for result in self.test_results:
            test_type = result['test']
            if test_type not in categories:
                categories[test_type] = []
            categories[test_type].append(result)
        
        for category, results in categories.items():
            print(f"\n{category.upper()}:")
            for result in results:
                status = "✅" if result['passed'] else "❌"
                if 'file' in result:
                    print(f"  {status} {result['file']}")
                else:
                    print(f"  {status} {category}")
                
                if not result['passed'] and 'error' in result:
                    print(f"     Error: {result['error']}")
        
        # Performance summary
        perf_results = [r for r in self.test_results if r['test'] == 'performance']
        if perf_results:
            print("\n⚡ Performance Summary:")
            for result in perf_results:
                print(f"  Processing time: {result['processing_time']}s")
                print(f"  Within limit: {'Yes' if result['passed'] else 'No'}")
        
        # Save report to file
        with open('test_report.json', 'w') as f:
            json.dump({
                'summary': {
                    'total_tests': total_tests,
                    'passed': passed_tests,
                    'failed': total_tests - passed_tests,
                    'success_rate': (passed_tests/total_tests)*100
                },
                'results': self.test_results
            }, f, indent=2)
        
        print("\n✅ Report saved to test_report.json")
        
        # Final verdict
        if passed_tests == total_tests:
            print("\n🎉 ALL TESTS PASSED! Ready for submission!")
        else:
            print("\n⚠️  Some tests failed. Please review and fix before submission.")

if __name__ == "__main__":
    tester = FinalTester()
    tester.run_all_tests()