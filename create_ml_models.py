#!/usr/bin/env python3
"""
Create ML models for heading classification using the existing PDFs as training data
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import json
import os
from pathlib import Path
from rule_engine import SmartRuleEngine
import fitz
import re

class MLModelTrainer:
    def __init__(self):
        self.feature_extractor = None
        self.classifier = None
        self.text_vectorizer = None
        self.scaler = None
        
    def create_training_data(self):
        """Create training data from existing PDFs and expected outputs"""
        print("Creating training data from existing PDFs...")
        
        rule_engine = SmartRuleEngine()
        training_data = []
        
        # Process each training PDF
        pdf_files = ['file01.pdf', 'file02.pdf', 'file03.pdf', 'file04.pdf', 'file05.pdf']
        
        for pdf_file in pdf_files:
            pdf_path = f'test_data/{pdf_file}'
            expected_path = f'output/expected_output/{pdf_file.replace(".pdf", ".json")}'
            
            if not os.path.exists(pdf_path) or not os.path.exists(expected_path):
                continue
                
            print(f"Processing {pdf_file}...")
            
            # Load expected output
            with open(expected_path, 'r', encoding='utf-8') as f:
                expected = json.load(f)
            
            expected_headings = {h['text'].strip().lower(): h for h in expected['outline']}
            
            # Extract all blocks with features from PDF
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):
                blocks = self._extract_blocks_with_features(page, page_num, rule_engine)
                
                for block in blocks:
                    text_lower = block['text'].strip().lower()
                    
                    # Label as heading or not based on expected output
                    is_heading = text_lower in expected_headings
                    heading_level = expected_headings.get(text_lower, {}).get('level', None) if is_heading else None
                    
                    block['is_heading'] = is_heading
                    block['heading_level'] = heading_level
                    training_data.append(block)
            
            doc.close()
        
        print(f"Created {len(training_data)} training samples")
        return training_data
    
    def _extract_blocks_with_features(self, page, page_num, rule_engine):
        """Extract blocks with comprehensive features"""
        blocks = []
        page_dict = page.get_text("dict")
        page_height = page_dict["height"]
        page_width = page_dict["width"]
        
        # Get font hierarchy for this page
        doc = page.parent
        font_hierarchy = rule_engine.font_analyzer.analyze(doc)
        
        for block_idx, block in enumerate(page_dict["blocks"]):
            if block["type"] != 0:  # Skip non-text
                continue
            
            text = self._extract_text_from_block(block)
            if not text.strip() or len(text) > 200:  # Skip empty or very long blocks
                continue
            
            # Extract comprehensive features
            features = {
                'text': text.strip(),
                'page_num': page_num,
                'block_idx': block_idx,
                
                # Position features
                'x_normalized': block["bbox"][0] / page_width,
                'y_normalized': block["bbox"][1] / page_height,
                'width_normalized': (block["bbox"][2] - block["bbox"][0]) / page_width,
                'height_normalized': (block["bbox"][3] - block["bbox"][1]) / page_height,
                
                # Text features
                'text_length': len(text),
                'word_count': len(text.split()),
                'is_uppercase': text.isupper(),
                'starts_with_number': bool(re.match(r'^\d+', text)),
                'has_punctuation': text.strip()[-1] in '.!?:' if text.strip() else False,
                'ends_with_colon': text.strip().endswith(':'),
                
                # Pattern features
                'is_question': bool(re.match(r'^(What|How|Why|Where|When)', text, re.IGNORECASE)),
                'is_numbered_list': bool(re.match(r'^\d+\.', text)),
                'is_alphabetic_list': bool(re.match(r'^[a-zA-Z]\.', text)),
                'has_chapter_keyword': bool(re.search(r'\b(chapter|section|part|appendix)\b', text, re.IGNORECASE)),
                
                # Font features
                'font_sizes': [],
                'is_bold': False,
                'is_italic': False,
                'font_names': set()
            }
            
            # Extract font information
            self._extract_font_features(block, features)
            
            # Calculate font statistics
            if features['font_sizes']:
                features['avg_font_size'] = np.mean(features['font_sizes'])
                features['max_font_size'] = max(features['font_sizes'])
                features['min_font_size'] = min(features['font_sizes'])
                features['font_size_variance'] = np.var(features['font_sizes'])
            else:
                features['avg_font_size'] = 0
                features['max_font_size'] = 0
                features['min_font_size'] = 0
                features['font_size_variance'] = 0
            
            features['font_variety'] = len(features['font_names'])
            
            # Relative font size features (compared to document)
            if font_hierarchy:
                body_size = font_hierarchy.get('body', 12)
                features['font_ratio_to_body'] = features['avg_font_size'] / body_size if body_size > 0 else 1
                features['is_larger_than_body'] = features['avg_font_size'] > body_size
            else:
                features['font_ratio_to_body'] = 1
                features['is_larger_than_body'] = False
            
            blocks.append(features)
        
        return blocks
    
    def _extract_font_features(self, block, features):
        """Extract font-related features from block"""
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                features['font_sizes'].append(span.get("size", 0))
                features['font_names'].add(span.get("font", ""))
                
                flags = span.get("flags", 0)
                if flags & 2**4:  # Bold
                    features['is_bold'] = True
                if flags & 2**1:  # Italic
                    features['is_italic'] = True
    
    def _extract_text_from_block(self, block):
        """Extract text from a PyMuPDF block"""
        text_parts = []
        for line in block.get("lines", []):
            line_text = ""
            for span in line.get("spans", []):
                span_text = span.get("text", "").strip()
                if span_text:
                    line_text += span_text + " "
            if line_text.strip():
                text_parts.append(line_text.strip())
        return " ".join(text_parts)
    
    def prepare_features(self, training_data):
        """Prepare features for ML training"""
        print("Preparing features for ML training...")
        
        # Extract text features using TF-IDF
        texts = [item['text'] for item in training_data]
        self.text_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        text_features = self.text_vectorizer.fit_transform(texts).toarray()
        
        # Extract numerical features
        numerical_features = []
        feature_names = [
            'x_normalized', 'y_normalized', 'width_normalized', 'height_normalized',
            'text_length', 'word_count', 'avg_font_size', 'max_font_size', 
            'font_size_variance', 'font_variety', 'font_ratio_to_body'
        ]
        
        boolean_features = [
            'is_uppercase', 'starts_with_number', 'has_punctuation', 'ends_with_colon',
            'is_question', 'is_numbered_list', 'is_alphabetic_list', 'has_chapter_keyword',
            'is_bold', 'is_italic', 'is_larger_than_body'
        ]
        
        for item in training_data:
            row = []
            # Add numerical features
            for feature in feature_names:
                row.append(item.get(feature, 0))
            
            # Add boolean features (convert to int)
            for feature in boolean_features:
                row.append(int(item.get(feature, False)))
            
            numerical_features.append(row)
        
        numerical_features = np.array(numerical_features)
        
        # Scale numerical features
        self.scaler = StandardScaler()
        numerical_features_scaled = self.scaler.fit_transform(numerical_features)
        
        # Combine text and numerical features
        combined_features = np.hstack([text_features, numerical_features_scaled])
        
        # Extract labels
        labels = [item['is_heading'] for item in training_data]
        
        return combined_features, labels
    
    def train_models(self):
        """Train the ML models"""
        print("Starting ML model training...")
        
        # Create training data
        training_data = self.create_training_data()
        
        if len(training_data) == 0:
            print("No training data found!")
            return False
        
        # Prepare features
        X, y = self.prepare_features(training_data)
        
        print(f"Training with {len(X)} samples, {X.shape[1]} features")
        print(f"Positive samples (headings): {sum(y)}")
        print(f"Negative samples (non-headings): {len(y) - sum(y)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train classifier
        print("Training Random Forest classifier...")
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'  # Handle imbalanced data
        )
        
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.classifier.score(X_train, y_train)
        test_score = self.classifier.score(X_test, y_test)
        
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Test accuracy: {test_score:.3f}")
        
        # Detailed evaluation
        y_pred = self.classifier.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return True
    
    def save_models(self):
        """Save the trained models"""
        print("Saving models...")
        
        os.makedirs('models', exist_ok=True)
        
        # Create a feature extractor that combines text vectorizer and scaler
        feature_extractor = {
            'text_vectorizer': self.text_vectorizer,
            'scaler': self.scaler,
            'feature_names': [
                'x_normalized', 'y_normalized', 'width_normalized', 'height_normalized',
                'text_length', 'word_count', 'avg_font_size', 'max_font_size', 
                'font_size_variance', 'font_variety', 'font_ratio_to_body'
            ],
            'boolean_features': [
                'is_uppercase', 'starts_with_number', 'has_punctuation', 'ends_with_colon',
                'is_question', 'is_numbered_list', 'is_alphabetic_list', 'has_chapter_keyword',
                'is_bold', 'is_italic', 'is_larger_than_body'
            ]
        }
        
        joblib.dump(feature_extractor, 'models/feature_extractor.pkl')
        joblib.dump(self.classifier, 'models/heading_classifier.pkl')
        
        print("Models saved successfully!")
        print("- models/feature_extractor.pkl")
        print("- models/heading_classifier.pkl")

if __name__ == "__main__":
    trainer = MLModelTrainer()
    
    if trainer.train_models():
        trainer.save_models()
        print("\nML models created successfully!")
    else:
        print("Failed to create ML models")
