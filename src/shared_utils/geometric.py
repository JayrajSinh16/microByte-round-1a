# geometric.py
from typing import List, Dict

class GeometricUtils:
    """Utilities for geometric calculations and spatial analysis"""
    
    @staticmethod
    def bboxes_overlap(bbox1: List, bbox2: List) -> bool:
        """Check if two bounding boxes overlap"""
        return not (bbox1[2] <= bbox2[0] or bbox2[2] <= bbox1[0] or 
                   bbox1[3] <= bbox2[1] or bbox2[3] <= bbox1[1])

    @staticmethod
    def is_block_in_table(block: Dict, table_areas: List[Dict]) -> bool:
        """Check if a block overlaps with any table area"""
        if not table_areas:
            return False
            
        block_bbox = block['bbox']
        for table in table_areas:
            if GeometricUtils.bboxes_overlap(block_bbox, table['bbox']):
                return True
        return False

    @staticmethod
    def calculate_line_bbox(line: Dict) -> List[float]:
        """Calculate bounding box for a line"""
        if not line.get("spans"):
            return [0, 0, 0, 0]
        
        x_coords = []
        y_coords = []
        
        for span in line["spans"]:
            bbox = span.get("bbox", [0, 0, 0, 0])
            x_coords.extend([bbox[0], bbox[2]])
            y_coords.extend([bbox[1], bbox[3]])
        
        if x_coords and y_coords:
            return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
        return [0, 0, 0, 0]