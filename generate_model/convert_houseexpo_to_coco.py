import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

class HouseExpoConverter:
    """Convert HouseExpo to COCO format for Detectron2"""
    
    def __init__(self):
        self.json_dir = Path("HouseExpo/HouseExpo/json")
        self.output_dir = Path("./detectron2_data")
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        # HouseExpo room categories mapped to our categories
        self.room_mapping = {
            'Bedroom': 'bedroom',
            'LivingRoom': 'living_room',
            'Kitchen': 'kitchen',
            'Bathroom': 'bathroom',
            'DiningRoom': 'dining_room',
            'Garage': 'garage',
            'Closet': 'closet',
            'Hallway': 'hallway',
            'Office': 'bedroom',  # Map office to bedroom
            'Library': 'bedroom',  # Map library to bedroom
            'Balcony': 'hallway',  # Map balcony to hallway
        }
        
        self.categories = [
            {'id': 1, 'name': 'bedroom'},
            {'id': 2, 'name': 'bathroom'},
            {'id': 3, 'name': 'kitchen'},
            {'id': 4, 'name': 'living_room'},
            {'id': 5, 'name': 'dining_room'},
            {'id': 6, 'name': 'garage'},
            {'id': 7, 'name': 'closet'},
            {'id': 8, 'name': 'hallway'},
        ]
        
        self.category_name_to_id = {cat['name']: cat['id'] for cat in self.categories}
    
    def load_house_json(self, json_path):
        """Load house JSON from HouseExpo"""
        with open(json_path, 'r') as f:
            return json.load(f)
    
    def create_floor_plan_image(self, house_data, house_id, size=512):
        """Create floor plan image from house data"""
        # Get house bounds
        bbox = house_data['bbox']
        min_x, min_y = bbox['min']
        max_x, max_y = bbox['max']
        
        # Calculate scale
        width = max_x - min_x
        height = max_y - min_y
        scale = min(size / width, size / height) * 0.9  # 90% to add padding
        
        # Create blank image
        img = np.ones((size, size, 3), dtype=np.uint8) * 255
        
        # Offset for centering
        offset_x = (size - width * scale) / 2
        offset_y = (size - height * scale) / 2
        
        def transform_point(x, y):
            """Transform point to image coordinates"""
            px = int((x - min_x) * scale + offset_x)
            py = int((max_y - y) * scale + offset_y)  # Flip Y
            return px, py
        
        # Draw rooms
        room_polygons = []
        
        for room_type, rooms in house_data.get('room_category', {}).items():
            if room_type not in self.room_mapping:
                continue
            
            mapped_type = self.room_mapping[room_type]
            category_id = self.category_name_to_id.get(mapped_type)
            
            if category_id is None:
                continue
            
            # rooms can be a list of bboxes or a single bbox
            if not isinstance(rooms, list):
                rooms = [rooms]
            
            for room_bbox in rooms:
                if len(room_bbox) == 4:
                    x1, y1, x2, y2 = room_bbox
                    
                    # Create rectangle polygon
                    pts = np.array([
                        transform_point(x1, y1),
                        transform_point(x2, y1),
                        transform_point(x2, y2),
                        transform_point(x1, y2)
                    ], dtype=np.int32)
                    
                    # Draw filled rectangle
                    cv2.fillPoly(img, [pts], (200, 200, 200))
                    cv2.polylines(img, [pts], True, (0, 0, 0), 2)
                    
                    room_polygons.append({
                        'category_id': category_id,
                        'polygon': pts
                    })
        
        # Draw walls (house outline)
        if 'verts' in house_data:
            verts = house_data['verts']
            if len(verts) > 2:
                wall_pts = np.array([transform_point(x, y) for x, y in verts], dtype=np.int32)
                cv2.polylines(img, [wall_pts], True, (0, 0, 0), 3)
        
        return img, room_polygons, scale, offset_x, offset_y, min_x, max_y
    
    def convert_to_coco(self, num_samples=1000):
        """Convert HouseExpo samples to COCO format"""
        
        json_files = list(self.json_dir.glob("*.json"))
        
        if not json_files:
            print(f"ERROR: No JSON files found in {self.json_dir}")
            return
        
        print(f"\nFound {len(json_files)} house plans")
        print(f"Converting {min(num_samples, len(json_files))} samples...")
        
        # Sample files
        import random
        random.seed(42)
        sampled_files = random.sample(json_files, min(num_samples, len(json_files)))
        
        coco_data = {
            'info': {
                'description': 'HouseExpo Floor Plans for Room Detection',
                'date_created': datetime.now().isoformat()
            },
            'images': [],
            'annotations': [],
            'categories': self.categories
        }
        
        image_id = 0
        annotation_id = 0
        
        for json_file in tqdm(sampled_files):
            try:
                # Load house data
                house_data = self.load_house_json(json_file)
                house_id = json_file.stem
                
                # Create floor plan image
                img, room_polygons, scale, offset_x, offset_y, min_x, max_y = \
                    self.create_floor_plan_image(house_data, house_id)
                
                if not room_polygons:
                    continue
                
                # Save image
                img_filename = f"{house_id}.png"
                img_path = self.images_dir / img_filename
                cv2.imwrite(str(img_path), img)
                
                # Add to COCO
                coco_data['images'].append({
                    'id': image_id,
                    'file_name': img_filename,
                    'width': img.shape[1],
                    'height': img.shape[0]
                })
                
                # Add annotations
                for room in room_polygons:
                    polygon = room['polygon']
                    
                    # Calculate bbox
                    x_coords = polygon[:, 0]
                    y_coords = polygon[:, 1]
                    x_min, x_max = x_coords.min(), x_coords.max()
                    y_min, y_max = y_coords.min(), y_coords.max()
                    
                    # Calculate area
                    area = cv2.contourArea(polygon)
                    
                    if area < 100:  # Skip tiny rooms
                        continue
                    
                    # Flatten polygon for COCO format
                    segmentation = polygon.flatten().tolist()
                    
                    coco_data['annotations'].append({
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': room['category_id'],
                        'segmentation': [segmentation],
                        'area': float(area),
                        'bbox': [int(x_min), int(y_min), 
                                int(x_max - x_min), int(y_max - y_min)],
                        'iscrowd': 0
                    })
                    annotation_id += 1
                
                image_id += 1
                
            except Exception as e:
                print(f"Error processing {json_file.name}: {e}")
                continue
        
        # Save COCO annotations
        output_file = self.output_dir / "annotations.json"
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"CONVERSION COMPLETE")
        print(f"{'='*60}")
        print(f"Images: {len(coco_data['images'])}")
        print(f"Annotations: {len(coco_data['annotations'])}")
        print(f"Output: {output_file.resolve()}")
        print(f"Images: {self.images_dir.resolve()}")
        print(f"{'='*60}")

def main():
    converter = HouseExpoConverter()
    converter.convert_to_coco(num_samples=2000)  # Use 2000 samples

if __name__ == "__main__":
    main()
