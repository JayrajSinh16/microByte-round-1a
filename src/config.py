# Final config.py with all optimizations
class Config:
    # Performance limits
    MAX_PROCESSING_TIME = 10
    MAX_MODEL_SIZE = 200
    
    # Detection thresholds
    TITLE_SIZE_RATIO = 1.5
    H1_SIZE_RATIO = 1.3
    H2_SIZE_RATIO = 1.15
    H3_SIZE_RATIO = 1.1
    
    # PDF Analysis thresholds
    MIN_TEXT_EXTRACTION_RATE = 0.5
    MAX_FONT_VARIETY = 15
    SCANNED_PAGE_THRESHOLD = 0.3
    
    # Performance settings
    ENABLE_PARALLEL = True
    PARALLEL_THRESHOLD = 10  # pages
    CACHE_SIZE = 128
    OCR_DPI = 1.5  # Balance quality/speed
    
    # Heading patterns
    HEADING_PATTERNS = {
        'number_patterns': [
            r'^\d+\.?\s+',           # 1. or 1
            r'^\d+\.\d+\.?\s+',      # 1.1. or 1.1
            r'^\d+\.\d+\.\d+\.?\s+'  # 1.1.1. or 1.1.1
        ],
        'keyword_patterns': [
            r'^(Chapter|CHAPTER|Section|SECTION)\s+\d+',
            r'^(Introduction|Conclusion|Abstract|References)',
            r'^(Background|Methodology|Results|Discussion)'
        ]
    }
    
    # Debug settings
    DEBUG = False
    LOG_LEVEL = 'INFO'
    SAVE_INTERMEDIATE = False