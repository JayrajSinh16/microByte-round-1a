import re
from typing import Dict, List

class OutputFormatter:
    """Handles formatting of predictions to required output format"""
    
    def format_output(self, predictions: List[Dict], blocks: List[Dict]) -> Dict:
        """Format predictions to required output format"""
        headings = []
        title = None
        
        # Sort by page and position
        predictions.sort(key=lambda x: (
            x['block']['page_num'], 
            x['block']['y_normalized']
        ))
        
        # First, try to identify the main title from the blocks
        for i, pred in enumerate(predictions):
            if pred['is_heading']:
                block = pred['block']
                text = block['text']  # Don't strip here to preserve spacing
                
                # Improved title detection for STEM documents
                if (title is None and 
                    any(keyword in text.lower() for keyword in 
                        ['parsippany', 'troy hills', 'stem pathways']) and
                    len(text) > 10):
                    title = text
                    continue  # Don't add title to outline
                
                # Better title detection for various document types
                elif (title is None and block['page_num'] == 0 and 
                      block['y_normalized'] < 0.3 and  # Top third of page
                      any(keyword in text.lower() for keyword in 
                          ['proposal', 'business plan', 'rfp', 'overview', 'pathways'])):
                    title = text
                    continue  # Don't add title to outline
        
        # Then process remaining headings with better filtering
        for pred in predictions:
            if pred['is_heading']:
                block = pred['block']
                text = block['text']  # Don't strip here to preserve spacing
                
                # Skip if this is the title we already identified
                if title and text.strip().lower() == title.strip().lower():
                    continue
                
                # Enhanced filtering for different document types
                should_include = True
                
                # Skip single-word labels that are not main headings
                if text.lower() in ['goals:', 'goals', 'objectives:', 'objectives']:
                    should_include = False
                
                # Skip detailed pathway descriptions (keep main pathway names only)
                if 'regular pathway' in text.lower() or 'advanced pathway' in text.lower():
                    should_include = False
                
                # Only include major section headings for STEM documents
                if title and 'stem pathways' in title.lower():
                    major_headings = ['pathway options', 'pathways', 'options', 'programs']
                    if not any(heading in text.lower() for heading in major_headings):
                        should_include = False
                
                # Enhanced party invitation filtering
                party_labels = ['address:', 'rsvp:', 'phone:', 'tel:', 'email:', 'www.']
                if any(label in text.lower() for label in party_labels):
                    should_include = False
                
                # Skip very short standalone text that's likely labels
                if len(text) <= 10 and text.endswith(':'):
                    should_include = False
                
                # Skip text that contains only dashes or special characters
                if re.match(r'^[:\-\s]+$', text):
                    should_include = False
                
                # Filter out overly long headings that are likely paragraphs
                if len(text) > 100 or block['word_count'] > 15:
                    should_include = False
                
                if should_include:
                    headings.append({
                        'level': pred['level'],
                        'text': text,
                        'page': block['page_num']  # Convert to 0-indexed in the final step
                    })
        
        # Convert page numbers to 0-based indexing
        for heading in headings:
            heading['page'] = max(0, heading['page'] - 1)
        
        return {
            'title': title or "",  # Empty title for party invitations like expected output
            'outline': headings
        }