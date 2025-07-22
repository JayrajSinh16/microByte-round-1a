# src/rule_engine/smart_rule_engine.py
import fitz
from typing import Dict
from config import Config
from src.shared_utils import PatternMatchingUtils, FontHierarchyAnalyzer
from .title_extractor import TitleExtractor
from .heading_extractor import HeadingExtractor
from .content_analyzer import ContentAnalyzer

class SmartRuleEngine:
    """High-performance rule-based heading extractor"""
    
    def __init__(self):
        self.heading_patterns = PatternMatchingUtils.compile_common_patterns()
        self.font_analyzer = FontHierarchyAnalyzer()
        self.title_extractor = TitleExtractor()
        self.heading_extractor = HeadingExtractor(self.heading_patterns)
        self.content_analyzer = ContentAnalyzer()
        
    def extract(self, pdf_path: str) -> Dict:
        doc = fitz.open(pdf_path)
        
        # Phase 1: Statistical font analysis
        font_hierarchy = self.font_analyzer.analyze(doc)
        
        # Phase 2: Detect document type early
        doc_type = self.content_analyzer.detect_document_type(doc)
        
        # Phase 3: Extract title (skip for invitation-like documents)
        if doc_type == 'invitation':
            title = ""  # Invitations should have empty title to avoid extracting decorative text
        else:
            title = self.title_extractor.extract_title(doc, font_hierarchy)
        
        # Phase 4: Extract headings (excluding the title)
        headings = self.heading_extractor.extract_headings(doc, font_hierarchy, title)
        
        doc.close()
        
        return {
            "title": title,
            "outline": headings
        }