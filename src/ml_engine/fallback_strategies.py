import re
from typing import Dict, List
from .block_extractor import BlockExtractor
from src.shared_utils import PDFTextUtils

class FallbackStrategies:
    """Handles fallback strategies for problematic PDFs"""
    
    def __init__(self):
        self.block_extractor = BlockExtractor()
    
    def process_scanned_pdf_fallback(self, doc) -> Dict:
        """Enhanced fallback for scanned PDFs without OCR"""
        print("🔧 ML Engine: Processing scanned PDF without OCR - using enhanced text extraction")
        # Try to extract any available text using PyMuPDF's text extraction
        # even from image-based PDFs
        all_blocks = []
        
        for page_num, page in enumerate(doc):
            # Try multiple extraction methods
            text_dict = page.get_text("dict")
            
            # Extract any text blocks that might be available
            for block in text_dict.get("blocks", []):
                if block.get("type") == 0:  # Text block
                    text = self.block_extractor.extract_text_from_block(block)
                    if text.strip():
                        all_blocks.append({
                            'text': text.strip(),
                            'page_num': page_num + 1,
                            'bbox': block.get("bbox", [0, 0, 0, 0])
                        })
        
        # If we found some text, process it
        if all_blocks:
            print(f"✅ ML Engine: Found {len(all_blocks)} text blocks in scanned PDF")
            return self._process_extracted_blocks(all_blocks)
        else:
            # Last resort: Check if this is a known problematic PDF pattern
            print("⚠️  ML Engine: No text blocks found in scanned PDF, trying fallback strategies")
            return self.handle_corrupted_pdf_fallback(doc)
    
    def handle_corrupted_pdf_fallback(self, doc) -> Dict:
        """Intelligent fallback for PDFs with extraction issues - no hardcoding"""
        print("🔧 ML Engine: Attempting intelligent fallback extraction...")
        
        # Try multiple extraction strategies
        strategies = [
            self._try_basic_text_extraction,
            self._try_metadata_based_extraction,
            self._try_layout_analysis_fallback,
            self._try_pattern_based_extraction
        ]
        
        for strategy in strategies:
            try:
                result = strategy(doc)
                if result and (result.get('title') or result.get('outline')):
                    print(f"✅ ML Engine Fallback: Successful with {strategy.__name__}")
                    return result
            except Exception as e:
                print(f"❌ ML Engine Fallback: Strategy {strategy.__name__} failed: {e}")
                continue
        
        # If all strategies fail, return minimal structure
        print("⚠️  ML Engine Fallback: All strategies failed, returning minimal structure")
        return {
            "title": "Document (Extraction Failed)",
            "outline": []
        }
    
    def _try_basic_text_extraction(self, doc) -> Dict:
        """Try basic text extraction with simple heuristics"""
        print("🔍 ML Engine Fallback: Trying basic text extraction strategy")
        all_text = ""
        headings = []
        
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                all_text += text + "\n"
                
                # Split into lines and look for heading patterns
                lines = text.split('\n')
                for line in lines:
                    line = line.strip()
                    if not line or len(line) > 100:
                        continue
                    
                    # Simple heading detection
                    if (line.isupper() and len(line.split()) <= 8) or \
                       re.match(r'^\d+\.?\s+[A-Z]', line) or \
                       line.endswith(':') and len(line.split()) <= 6:
                        headings.append({
                            'level': 'H2',
                            'text': line,
                            'page': page_num
                        })
        
        # Try to extract title from metadata or first heading
        title = doc.metadata.get('title', '')
        if not title and headings:
            title = headings[0]['text']
            headings = headings[1:]
        
        return {
            'title': title or "Document",
            'outline': headings[:10]  # Limit to first 10 headings
        }
    
    def _try_metadata_based_extraction(self, doc) -> Dict:
        """Try extraction based on PDF metadata and structure"""
        print("🔍 ML Engine Fallback: Trying metadata-based extraction strategy")
        metadata = doc.metadata
        
        title = metadata.get('title', '') or metadata.get('subject', '')
        author = metadata.get('author', '')
        
        # Use outline/bookmarks if available
        outline = doc.get_toc()
        headings = []
        
        if outline:
            for level, title_text, page_num in outline:
                if level <= 3:  # Only include first 3 levels
                    headings.append({
                        'level': f'H{min(level, 3)}',
                        'text': title_text,
                        'page': max(0, page_num - 1)  # Convert to 0-based
                    })
        
        return {
            'title': title or "Document",
            'outline': headings
        }
    
    def _try_layout_analysis_fallback(self, doc) -> Dict:
        """Try layout-based analysis for structure detection"""
        print("🔍 ML Engine Fallback: Trying layout-based analysis strategy")
        all_blocks = []
        
        # Extract blocks with basic layout info
        for page_num, page in enumerate(doc):
            try:
                blocks = page.get_text("dict").get("blocks", [])
                for block in blocks:
                    if block.get("type") == 0:  # Text block
                        text = self.block_extractor.extract_text_from_block(block)
                        if text.strip() and len(text) < 200:
                            bbox = block.get("bbox", [0, 0, 0, 0])
                            all_blocks.append({
                                'text': text.strip(),
                                'page': page_num,
                                'y_pos': bbox[1],
                                'height': bbox[3] - bbox[1],
                                'is_top_of_page': bbox[1] < 100
                            })
            except:
                continue
        
        if not all_blocks:
            return {'title': '', 'outline': []}
        
        # Sort by page and position
        all_blocks.sort(key=lambda x: (x['page'], x['y_pos']))
        
        # Simple heading detection based on position and content
        headings = []
        title = None
        
        for block in all_blocks:
            text = block['text']
            
            # Skip very long text (paragraphs)
            if len(text.split()) > 20:
                continue
            
            # Title detection (first significant text on first page)
            if not title and block['page'] == 0 and block['is_top_of_page']:
                if len(text.split()) >= 3:
                    title = text
                    continue
            
            # Heading detection
            is_heading = False
            
            if block['is_top_of_page']:  # Top of page
                is_heading = True
            elif text.isupper() and len(text.split()) <= 8:  # All caps
                is_heading = True
            elif re.match(r'^\d+\.?\s+[A-Z]', text):  # Numbered
                is_heading = True
            elif text.endswith(':'):  # Ends with colon
                is_heading = True
            
            if is_heading:
                headings.append({
                    'level': 'H2',
                    'text': text,
                    'page': block['page']
                })
        
        return {
            'title': title or "Document",
            'outline': headings[:15]  # Limit headings
        }
    
    def _try_pattern_based_extraction(self, doc) -> Dict:
        """Try pattern-based extraction using common document patterns"""
        print("🔍 ML Engine Fallback: Trying pattern-based extraction strategy")
        all_text = ""
        
        # Collect all text
        for page in doc:
            all_text += page.get_text() + "\n"
        
        if not all_text.strip():
            return {'title': '', 'outline': []}
        
        lines = [line.strip() for line in all_text.split('\n') if line.strip()]
        
        headings = []
        title = None
        
        # Common patterns for different document types
        patterns = [
            (r'^(CHAPTER|Chapter)\s+\d+', 'H1'),
            (r'^(SECTION|Section)\s+\d+', 'H2'),
            (r'^\d+\.?\s+[A-Z][a-zA-Z\s]{3,30}$', 'H2'),
            (r'^[A-Z][A-Z\s]{5,30}$', 'H1'),  # All caps headings
            (r'^(Introduction|Conclusion|Summary|Background|Methodology)', 'H2'),
            (r'^(Abstract|References|Appendix)', 'H1'),
            (r'^[a-zA-Z\s]{3,50}:$', 'H3'),  # Ends with colon
            (r'^(What|How|Why|Where|When)\s+.{5,50}\??$', 'H3'),  # Questions
        ]
        
        for i, line in enumerate(lines[:100]):  # Check first 100 lines
            # Title detection
            if not title and i < 5 and len(line.split()) >= 3:
                title = line
                continue
            
            # Skip very long lines
            if len(line) > 100:
                continue
            
            # Apply patterns
            for pattern, level in patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    # Estimate page number (rough approximation)
                    page_num = min(i // 20, len(doc) - 1)
                    
                    headings.append({
                        'level': level,
                        'text': line,
                        'page': page_num
                    })
                    break
        
        return {
            'title': title or "Document",
            'outline': headings[:20]  # Limit headings
        }
    
    def _process_extracted_blocks(self, blocks: List[Dict]) -> Dict:
        """Process extracted text blocks to find headings"""
        headings = []
        title = None
        
        # Sort blocks by page and position
        blocks.sort(key=lambda b: (b['page_num'], b['bbox'][1] if b['bbox'] else 0))
        
        for i, block in enumerate(blocks):
            text = block['text']
            
            # Skip very long texts (likely paragraphs)
            if len(text.split()) > 20:
                continue
            
            # Heading detection heuristics
            is_heading = False
            
            # Pattern-based detection
            if re.match(r'^\d+\.?\s+\w+', text):  # Numbered headings
                is_heading = True
            elif text.isupper() and len(text.split()) <= 8:  # All caps, short
                is_heading = True
            elif text.endswith(':') and len(text.split()) <= 6:  # Ends with colon
                is_heading = True
            elif re.match(r'^(What|How|Why|Where|When|Goals?|Objectives?)', text, re.IGNORECASE):
                is_heading = True
            
            if is_heading:
                if i == 0 and not title:
                    title = text
                else:
                    headings.append({
                        'level': 'H1',  # Default level
                        'text': text,
                        'page': block['page_num']
                    })
        
        return {
            "title": title or "Document",
            "outline": headings
        }