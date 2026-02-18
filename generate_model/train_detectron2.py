import torch
import detectron2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from pathlib import Path
import json

class FloorPlanTrainer:
    """Train Detectron2 model for floor plan room detection"""
    
    def __init__(self):
        self.data_dir = Path("detectron2_data")
        self.output_dir = Path("detectron2_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Register dataset
        self.register_dataset()
        
        # Setup config
        self.cfg = self.setup_config()
    
    def register_dataset(self):
        """Register floor plan dataset with Detectron2"""
        print("Registering dataset...")
        
        # Check if annotations exist
        annotations_file = self.data_dir / "annotations.json"
        if not annotations_file.exists():
            print(f"ERROR: Annotations not found at {annotations_file}")
            print("Run annotate_detectron2.py first!")
            exit(1)
        
        # Load annotations to check
        with open(annotations_file, 'r') as f:
            data = json.load(f)
        
        print(f"  Images: {len(data['images'])}")
        print(f"  Annotations: {len(data['annotations'])}")
        print(f"  Categories: {len(data['categories'])}")
        
        if len(data['images']) < 10:
            print("WARNING: Less than 10 images annotated. Need at least 20 for good results.")
        
        # Register
        register_coco_instances(
            "floorplan_train",
            {},
            str(annotations_file),
            str(self.data_dir / "images")
        )
        
        # Set metadata
        MetadataCatalog.get("floorplan_train").thing_classes = [
            cat['name'] for cat in data['categories']
        ]
        
        print("✓ Dataset registered\n")
    
    def setup_config(self):
        """Setup Detectron2 config"""
        cfg = get_cfg()
        
        # Use Mask R-CNN with ResNet-50 backbone
        cfg.merge_from_file(
            model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        )
        
        cfg.DATASETS.TRAIN = ("floorplan_train",)
        cfg.DATASETS.TEST = ()
        
        cfg.DATALOADER.NUM_WORKERS = 2
        
        # Start from pre-trained COCO weights
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
        
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.00025
        cfg.SOLVER.MAX_ITER = 3000  # Adjust based on dataset size
        cfg.SOLVER.STEPS = []  # No learning rate decay
        
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9  # 9 room types
        
        cfg.OUTPUT_DIR = str(self.output_dir)
        
        return cfg
    
    def train(self):
        """Train the model"""
        print("="*60)
        print("TRAINING DETECTRON2 MODEL")
        print("="*60)
        print(f"Output directory: {self.output_dir.resolve()}")
        print(f"Max iterations: {self.cfg.SOLVER.MAX_ITER}")
        print(f"Learning rate: {self.cfg.SOLVER.BASE_LR}")
        print("="*60 + "\n")
        
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"Model saved to: {self.output_dir / 'model_final.pth'}")

def main():
    # Check CUDA
    print(f"PyTorch version: {torch.__version__}")
    print(f"Detectron2 version: {detectron2.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}\n")
    else:
        print("WARNING: No CUDA GPU detected. Training will be slow on CPU.\n")
    
    trainer = FloorPlanTrainer()
    trainer.train()

if __name__ == "__main__":
    main()
