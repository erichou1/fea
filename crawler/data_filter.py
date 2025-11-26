import os
import shutil
import torch
import clip
import hashlib
import re
from PIL import Image
from tqdm import tqdm
import numpy as np

# ==========================================
# 1. CONFIGURATION (STRICT MODE)
# ==========================================

# SENSITIVITY KNOB
# 0.5 = Loose (Allows some mistakes)
# 0.85 = Strict (Discards anything ambiguous)
CONFIDENCE_THRESHOLD = 0.85 

# INPUT / OUTPUT PATHS
INPUT_DIRS = ["./floor_plans", "./house_images"] 
TRASH_DIR = "./review"
FINAL_DIR = "./final_clean"

os.makedirs(TRASH_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

# SETUP DEVICE
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device} | Sensitivity: {CONFIDENCE_THRESHOLD}")
model, preprocess = clip.load("ViT-B/32", device=device)

# --- AGGRESSIVE PROMPT ENGINEERING ---
PROMPT_ENSEMBLE = {
    "floorplan": [
        "top-down view of a room layout",
        "black and white architectural blueprint",
        "overhead floor plan schematic",
        "cad drawing of walls from above",
        "technical drawing of a house map"
    ],
    "exterior": [
        "photo of a whole house from the street",
        "architectural elevation drawing of a house facade",
        "exterior view of a residential building",
        "front view line drawing of a home structure",
        "3d render of a house exterior"
    ],
    "trash": [
        # Explicit Interiors (The leak source)
        "interior photo of a living room with sofa",
        "interior photo of a kitchen with cabinets",
        "interior photo of a bedroom with bed",
        "inside view of a bathroom",
        "real estate photo of a hallway",
        "view looking out a window from inside",
        "close up of furniture or decor",
        # Standard Trash
        "text document or screenshot", 
        "blurry or pixelated image", 
        "company logo or watermark"
    ]
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def compute_dhash(image, hash_size=8):
    """ Perceptual Hashing to find duplicates """
    image = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(image.getdata())
    difference = []
    for row in range(hash_size):
        for col in range(hash_size):
            pixel_left = image.getpixel((col, row))
            pixel_right = image.getpixel((col + 1, row))
            difference.append(pixel_left > pixel_right)
    decimal_value = 0
    for index, value in enumerate(difference):
        if value: decimal_value += 2**index
    return decimal_value

def is_duplicate(new_hash, seen_hashes, threshold=4):
    """ Checks if image is similar to one already processed """
    for seen_h in seen_hashes:
        dist = bin(new_hash ^ seen_h).count('1')
        if dist <= threshold:
            return True
    return False

def extract_house_id(filename):
    match = re.match(r"^([a-zA-Z0-9]+)_", filename)
    if match: return match.group(1)
    return hashlib.md5(filename.encode()).hexdigest()[:8]

# ==========================================
# 3. PRE-COMPUTE PROMPTS
# ==========================================
print("Building Aggressive Prompt Ensembles...")
encoded_prompts = {}
with torch.no_grad():
    for cat, prompts in PROMPT_ENSEMBLE.items():
        tok = clip.tokenize(prompts).to(device)
        emb = model.encode_text(tok)
        emb /= emb.norm(dim=-1, keepdim=True)
        encoded_prompts[cat] = emb.mean(dim=0)
        encoded_prompts[cat] /= encoded_prompts[cat].norm()

text_features_stack = torch.stack([encoded_prompts["floorplan"], 
                                   encoded_prompts["exterior"], 
                                   encoded_prompts["trash"]])

# ==========================================
# 4. EXECUTION
# ==========================================

def process_and_clean():
    seen_hashes = set()
    id_counters = {} 
    
    # Gather files
    all_files = []
    for d in INPUT_DIRS:
        if os.path.exists(d):
            files = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(('.png','.jpg','.jpeg','.webp'))]
            all_files.extend(files)
    
    print(f"Found {len(all_files)} images. Starting Strict Filter...")

    stats = {"valid": 0, "trash": 0, "duplicate": 0, "low_conf": 0}

    for filepath in tqdm(all_files):
        try:
            pil_img = Image.open(filepath)
            
            # 1. DUPLICATE CHECK
            img_hash = compute_dhash(pil_img)
            if is_duplicate(img_hash, seen_hashes): 
                stats["duplicate"] += 1
                continue
            seen_hashes.add(img_hash)

            # 2. CLASSIFY
            image_input = preprocess(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_features @ text_features_stack.T).softmax(dim=-1)
                probs = similarity.cpu().numpy()[0]
            
            pred_idx = probs.argmax()
            confidence = probs[pred_idx]

            # 3. FILTER LOGIC
            
            # Case A: Explicit Trash
            if pred_idx == 2: 
                shutil.copy(filepath, os.path.join(TRASH_DIR, "trash_" + os.path.basename(filepath)))
                stats["trash"] += 1
                continue

            # Case B: Low Confidence (The "Ambiguous" Case)
            if confidence < CONFIDENCE_THRESHOLD:
                # If the model isn't sure, assume it's bad data
                shutil.copy(filepath, os.path.join(TRASH_DIR, "ambiguous_" + os.path.basename(filepath)))
                stats["low_conf"] += 1
                continue

            # Case C: Valid Data
            class_name = "floorplan" if pred_idx == 0 else "exterior"
            
            # Rename and Save
            house_id = extract_house_id(os.path.basename(filepath))
            if house_id not in id_counters: id_counters[house_id] = {"floorplan": 0, "exterior": 0}
            id_counters[house_id][class_name] += 1
            
            new_name = f"{house_id}_{class_name}_{id_counters[house_id][class_name]:02d}.png"
            pil_img.save(os.path.join(FINAL_DIR, new_name), "PNG")
            stats["valid"] += 1

        except Exception as e:
            print(f"Error on {filepath}: {e}")

    print("\n✅ Strict Processing Complete.")
    print(f"   - Kept (Valid):      {stats['valid']}")
    print(f"   - Removed (Trash):   {stats['trash']}")
    print(f"   - Removed (Ambiguous): {stats['low_conf']} (Confidence < {CONFIDENCE_THRESHOLD})")
    print(f"   - Removed (Dupes):   {stats['duplicate']}")

if __name__ == "__main__":
    process_and_clean()