# content_analyzer.py
from typing import Dict
from src.shared_utils import DocumentAnalysisUtils

class ContentAnalyzer:
    """Handles document type detection and content analysis"""
    
    def detect_document_type(self, doc) -> str:
        """Detect the type of document to apply specialized extraction rules"""
        return DocumentAnalysisUtils.detect_document_type(doc)