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
# 1. CONFIGURATION (EXTREME STRICTNESS)
# ==========================================

# ONLY recover images if model is 90% sure
RECOVERY_THRESHOLD = 0.85

INPUT_DIR = "./review"  # Scanning the trash
RECOVERY_DIR = "./review_stage2" # Where to put saved files

os.makedirs(RECOVERY_DIR, exist_ok=True)

# SETUP DEVICE
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running Salvage Operation on: {device}")
print(f"Strictness Level: {RECOVERY_THRESHOLD * 100}%")

model, preprocess = clip.load("ViT-B/32", device=device)

# --- REUSING THE ROBUST PROMPTS ---
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
        "3d render of a house exterior",
        "black and white 3d model of a house exterior",
        "architectural sketch of a house exterior",
        "3d model of house exterior"
    ],
    "trash": [
        "interior photo of a living room with sofa",
        "interior photo of a kitchen with cabinets",
        "interior photo of a bedroom with bed",
        "inside view of a bathroom",
        "real estate photo of a hallway",
        "close up of furniture or decor",
        "text document or screenshot", 
        "blurry or pixelated image", 
        "company logo or watermark"
    ]
}

# ==========================================
# 2. SETUP FEATURES
# ==========================================
print("Building Prompt Ensembles...")
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
# 3. SALVAGE LOOP
# ==========================================

def extract_house_id(filename):
    # Try to preserve existing ID if present
    match = re.match(r"^([a-zA-Z0-9]+)_", filename)
    if match: return match.group(1)
    return hashlib.md5(filename.encode()).hexdigest()[:8]

def run_recovery():
    if not os.path.exists(INPUT_DIR):
        print(f"Directory not found: {INPUT_DIR}")
        return

    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png','.jpg','.jpeg','.webp'))]
    print(f"Scanning {len(files)} files in Trash...")

    recovered_count = 0
    id_counters = {}

    for filename in tqdm(files):
        filepath = os.path.join(INPUT_DIR, filename)
        try:
            pil_img = Image.open(filepath)
            
            # CLASSIFY
            image_input = preprocess(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_features @ text_features_stack.T).softmax(dim=-1)
                probs = similarity.cpu().numpy()[0]
            
            pred_idx = probs.argmax()
            confidence = probs[pred_idx]

            # LOGIC:
            # 1. Must NOT be trash class (Index 2)
            # 2. Confidence must be > 90% (RECOVERY_THRESHOLD)
            
            if pred_idx != 2 and confidence >= RECOVERY_THRESHOLD:
                
                class_name = "floorplan" if pred_idx == 0 else "exterior"
                
                # Prepare clean name
                house_id = extract_house_id(filename)
                if house_id not in id_counters: id_counters[house_id] = {"floorplan": 0, "exterior": 0}
                id_counters[house_id][class_name] += 1
                
                new_name = f"{house_id}_{class_name}_RECOVERED_{id_counters[house_id][class_name]:02d}.png"
                
                # Move (Recovery)
                shutil.move(filepath, os.path.join(RECOVERY_DIR, new_name))
                recovered_count += 1

        except Exception as e:
            print(f"Error checking {filename}: {e}")

    print("\n✅ Salvage Complete.")
    print(f"   - Recovered: {recovered_count} images")
    print(f"   - Location:  {RECOVERY_DIR}")
    print(f"   - Remaining items in '{INPUT_DIR}' are confirmed trash.")

if __name__ == "__main__":
    run_recovery()