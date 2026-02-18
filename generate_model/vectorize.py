import sys
sys.path.insert(0, './mmdetection')

from mmdet.apis import init_detector, inference_detector
import mmcv
import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

class FloorPlanBatchDetector:
    """Batch inference using pre-trained MMDetection model"""
    
    def __init__(self, config_path, checkpoint_path):
        self.model = init_detector(config_path, checkpoint_path, device='cuda:0')
        
        self.output_dir = Path("../cubicasa_output")
        self.output_dir.mkdir(exist_ok=True)
        
        self.debug_dir = self.output_dir / "debug"
        self.debug_dir.mkdir(exist_ok=True)
        
        # Class names: [walls, rooms]
        self.classes = ['wall', 'room']
    
    def detect_rooms_and_walls(self, img_path):
        """Run detection on single image"""
        img = mmcv.imread(str(img_path))
        result = inference_detector(self.model, img)
        
        # result[0] = walls (bboxes)
        # result[1] = rooms (bboxes)
        walls_bboxes = result[0] if len(result[0]) > 0 else []
        rooms_bboxes = result[1] if len(result[1]) > 0 else []
        
        return walls_bboxes, rooms_bboxes, img
    
    def bbox_to_polygon(self, bbox):
        """Convert bbox to polygon vertices"""
        x1, y1, x2, y2, score = bbox
        return [
            [int(x1), int(y1)],
            [int(x2), int(y1)],
            [int(x2), int(y2)],
            [int(x1), int(y2)]
        ]
    
    def classify_room_by_position(self, bbox, img_w, img_h):
        """Classify room type by position and size"""
        x1, y1, x2, y2, score = bbox
        
        width = x2 - x1
        height = y2 - y1
        area = width * height
        area_pct = area / (img_w * img_h)
        
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Position ratios
        x_ratio = center_x / img_w
        y_ratio = center_y / img_h
        
        # Bottom/side positions
        is_bottom = (y2 / img_h) > 0.65
        is_left = x_ratio < 0.3
        is_right = x_ratio > 0.7
        
        # Classify by area and position
        if area_pct > 0.08 and (is_bottom or is_left or is_right):
            return 'garage'
        elif area_pct < 0.015:
            return 'closet'
        elif area_pct < 0.04:
            return 'bathroom'
        elif area_pct < 0.10:
            return 'bedroom'
        elif area_pct < 0.18:
            return 'kitchen'
        else:
            return 'living_room'
    
    def save_json(self, rooms_bboxes, img_shape, house_id):
        """Save Plan2Scene JSON format"""
        h, w = img_shape[:2]
        
        rooms_data = []
        
        for idx, bbox in enumerate(rooms_bboxes):
            if len(bbox) < 5:
                continue
            
            score = float(bbox[4])
            if score < 0.5:  # Confidence threshold
                continue
            
            vertices = self.bbox_to_polygon(bbox)
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            room_type = self.classify_room_by_position(bbox, w, h)
            
            rooms_data.append({
                'id': f'room_{idx}',
                'type': room_type,
                'vertices': vertices,
                'area': float(area),
                'confidence': float(score)
            })
        
        # Sort by area
        rooms_data.sort(key=lambda x: x['area'], reverse=True)
        
        plan2scene_data = {
            'id': house_id,
            'rooms': rooms_data,
            'walls': [],
            'metadata': {'width': w, 'height': h, 'scale': 1.0, 'unit': 'pixels'}
        }
        
        # Add walls
        wall_id = 0
        for room in rooms_data:
            vertices = room['vertices']
            for j in range(len(vertices)):
                plan2scene_data['walls'].append({
                    'id': f'wall_{wall_id}',
                    'start': vertices[j],
                    'end': vertices[(j+1) % len(vertices)],
                    'room_id': room['id']
                })
                wall_id += 1
        
        output_path = self.output_dir / f'{house_id}.json'
        with open(output_path, 'w') as f:
            json.dump(plan2scene_data, f, indent=2)
    
    def save_visualization(self, img, walls_bboxes, rooms_bboxes, house_id):
        """Save debug visualization"""
        vis_img = img.copy()
        
        # Draw walls in red
        for bbox in walls_bboxes:
            if len(bbox) < 5 or bbox[4] < 0.5:
                continue
            x1, y1, x2, y2 = map(int, bbox[:4])
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Draw rooms in different colors
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), 
                 (255,0,255), (0,255,255), (255,128,0), (128,255,0)]
        
        for idx, bbox in enumerate(rooms_bboxes):
            if len(bbox) < 5 or bbox[4] < 0.5:
                continue
            
            x1, y1, x2, y2, score = map(int, bbox[:5])
            color = colors[idx % len(colors)]
            
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 3)
            
            # Add label
            label = f"Room {idx}: {score:.2f}"
            cv2.putText(vis_img, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        vis_path = self.debug_dir / f'{house_id}_detected.png'
        cv2.imwrite(str(vis_path), vis_img)
    
    def process_all(self, input_dir="../../floor_plans2"):
        """Process all floor plans"""
        input_path = Path(input_dir)
        images = list(input_path.glob("*.jpg"))
        
        print(f"{'='*60}")
        print("MMDETECTION FLOOR PLAN DETECTION")
        print(f"{'='*60}")
        print(f"Processing {len(images)} floor plans...\n")
        
        success = 0
        for img_path in tqdm(images, desc="Processing"):
            try:
                house_id = img_path.stem
                
                walls, rooms, img = self.detect_rooms_and_walls(img_path)
                
                if len(rooms) > 0:
                    self.save_json(rooms, img.shape, house_id)
                    self.save_visualization(img, walls, rooms, house_id)
                    success += 1
                
            except Exception as e:
                print(f"Error on {img_path.name}: {e}")
        
        print(f"\n{'='*60}")
        print(f"COMPLETE: {success}/{len(images)}")
        print(f"Output: {self.output_dir.resolve()}")
        print(f"{'='*60}\n")

def main():
    # Paths
    config_path = 'configs/cascade_swin.py'
    checkpoint_path = 'weights/cascade_swin.pth'
    
    detector = FloorPlanBatchDetector(config_path, checkpoint_path)
    detector.process_all()

if __name__ == "__main__":
    main()
