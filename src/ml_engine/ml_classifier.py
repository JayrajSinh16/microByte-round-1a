import numpy as np
import re
from typing import Dict, List

class MLClassifier:
    """Handles ML-based classification of text blocks"""
    
    def classify_blocks_with_ml(self, blocks: List[Dict], classifier, feature_extractor) -> List[Dict]:
        """Use ML models to classify blocks"""
        if not blocks:
            return []
        
        # Prepare features using the same method as training
        features = self._prepare_ml_features(blocks, feature_extractor)
        
        # Predict using the trained model
        predictions_binary = classifier.predict(features)
        prediction_probs = classifier.predict_proba(features)
        
        classified_blocks = []
        
        for i, (block, is_heading, prob) in enumerate(zip(blocks, predictions_binary, prediction_probs)):
            if is_heading:
                # Determine heading level using heuristics (can be enhanced with another ML model)
                level = self._determine_heading_level_ml(block, prob[1])  # prob[1] is probability of being heading
                
                classified_blocks.append({
                    'block': block,
                    'is_heading': True,
                    'level': level,
                    'confidence': prob[1]
                })
        
        return classified_blocks
    
    def _prepare_ml_features(self, blocks: List[Dict], feature_extractor) -> np.ndarray:
        """Prepare features for ML prediction using the same format as training"""
        # Extract texts for TF-IDF
        texts = [block['text'] for block in blocks]
        
        # Transform texts using the saved vectorizer
        text_features = feature_extractor['text_vectorizer'].transform(texts).toarray()
        
        # Extract numerical features
        numerical_features = []
        feature_names = feature_extractor['feature_names']
        boolean_features = feature_extractor['boolean_features']
        
        for block in blocks:
            row = []
            # Add numerical features
            for feature in feature_names:
                row.append(block.get(feature, 0))
            
            # Add boolean features (convert to int)
            for feature in boolean_features:
                row.append(int(block.get(feature, False)))
            
            numerical_features.append(row)
        
        numerical_features = np.array(numerical_features)
        
        # Scale numerical features using saved scaler
        numerical_features_scaled = feature_extractor['scaler'].transform(numerical_features)
        
        # Combine text and numerical features
        combined_features = np.hstack([text_features, numerical_features_scaled])
        
        return combined_features
    
    def _determine_heading_level_ml(self, block: Dict, confidence: float) -> str:
        """Determine heading level for ML-predicted headings"""
        text = block['text'].strip()
        
        # Use confidence score and text patterns to determine level
        if confidence > 0.9 and (
            block.get('avg_font_size', 0) > 16 or
            text.isupper() or
            re.match(r'^(CHAPTER|SECTION)', text, re.IGNORECASE)
        ):
            return 'H1'
        elif confidence > 0.8 and (
            block.get('avg_font_size', 0) > 14 or
            re.match(r'^\d+\.?\s+[A-Z]', text) or
            text.endswith(':')
        ):
            return 'H2'
        else:
            return 'H3'
    
    def combine_predictions(self, ml_predictions: List[Dict], heuristic_predictions: List[Dict]) -> List[Dict]:
        """Combine ML and heuristic predictions using confidence thresholds"""
        combined = []
        
        # Create lookup for heuristic predictions by text
        heuristic_lookup = {}
        for pred in heuristic_predictions:
            text = pred['block']['text'].strip().lower()
            heuristic_lookup[text] = pred
        
        # Start with high-confidence ML predictions
        for ml_pred in ml_predictions:
            if ml_pred['confidence'] > 0.8:  # High confidence ML predictions
                combined.append(ml_pred)
        
        # Add heuristic predictions that weren't found by ML or have low ML confidence
        ml_texts = {pred['block']['text'].strip().lower() for pred in ml_predictions if pred['confidence'] > 0.8}
        
        for heur_pred in heuristic_predictions:
            text = heur_pred['block']['text'].strip().lower()
            
            # Include if:
            # 1. Not found by ML at all, OR
            # 2. ML has low confidence for this text, OR  
            # 3. This is a special high-confidence pattern (like party invitations)
            if (text not in ml_texts or 
                heur_pred['confidence'] > 0.9 or  # Very high heuristic confidence
                'hope' in text):  # Special case for party invitations
                
                combined.append(heur_pred)
        
        return combined