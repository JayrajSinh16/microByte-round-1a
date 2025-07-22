"""
Shared utilities for PDF processing across different engines.
Contains common functionality used by both RuleEngine and MLEngine.
"""

# Import all utility classes for backward compatibility
from .pdf_text import PDFTextUtils
from .geometric import GeometricUtils
from .document_analysis import DocumentAnalysisUtils
from .table_detection import TableDetectionUtils
from .toc_detection import TOCDetectionUtils
from .text_normalization import TextNormalizationUtils
from .font_hierarchy import FontHierarchyAnalyzer
from .pattern_matching import PatternMatchingUtils

__all__ = [
    'PDFTextUtils',
    'GeometricUtils',
    'DocumentAnalysisUtils',
    'TableDetectionUtils',
    'TOCDetectionUtils',
    'TextNormalizationUtils',
    'FontHierarchyAnalyzer',
    'PatternMatchingUtils'
]