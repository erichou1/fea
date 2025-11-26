import os
import time
import requests
import hashlib
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from PIL import Image
from io import BytesIO
import torch
import clip  # Using the OpenAI CLIP library matching your filter
import numpy as np

# ==========================================
# 1. CONFIGURATION
# ==========================================

# SENSITIVITY KNOB (Strict Mode)
CONFIDENCE_THRESHOLD = 0.85

FLOOR_PLAN_DIR = "floor_plans2"
HOUSE_IMAGE_DIR = "house_images2"
TARGET_PLAN_COUNT = 250

# Setup Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device} | Sensitivity: {CONFIDENCE_THRESHOLD}")

# Load CLIP Model
print("Loading CLIP model...")
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
        # Explicit Interiors
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
# 2. PRE-COMPUTE TEXT FEATURES (Optimization)
# ==========================================
print("Building Prompt Ensembles...")
encoded_prompts = {}
with torch.no_grad():
    for cat, prompts in PROMPT_ENSEMBLE.items():
        tok = clip.tokenize(prompts).to(device)
        emb = model.encode_text(tok)
        emb /= emb.norm(dim=-1, keepdim=True)
        # Average the embeddings for the ensemble
        encoded_prompts[cat] = emb.mean(dim=0)
        encoded_prompts[cat] /= encoded_prompts[cat].norm()

# Stack: [0]=floorplan, [1]=exterior, [2]=trash
text_features_stack = torch.stack([encoded_prompts["floorplan"], 
                                   encoded_prompts["exterior"], 
                                   encoded_prompts["trash"]])

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

seen_image_hashes = set()

def compute_dhash(image, hash_size=8):
    """ 
    Robust Perceptual Hashing (matches your filter script).
    """
    try:
        # Convert to grayscale and resize
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
    except Exception as e:
        print(f"Hashing error: {e}")
        return None

def is_duplicate(new_hash, seen_hashes, threshold=4):
    """ Checks Hamming distance for similarity """
    if new_hash is None: return True
    for seen_h in seen_hashes:
        dist = bin(new_hash ^ seen_h).count('1')
        if dist <= threshold:
            return True
    return False

def classify_image_strict(image_bytes):
    """
    Returns: "FLOOR_PLAN", "HOUSE_EXTERIOR", or None (if trash/ambiguous)
    """
    try:
        # Preprocess Image
        pil_image = Image.open(BytesIO(image_bytes))
        image_input = preprocess(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Dot product with pre-computed text features
            similarity = (100.0 * image_features @ text_features_stack.T).softmax(dim=-1)
            probs = similarity.cpu().numpy()[0]
        
        pred_idx = probs.argmax()
        confidence = probs[pred_idx]

        # LOGIC MAPPING
        # Index 0 = floorplan
        # Index 1 = exterior
        # Index 2 = trash

        # 1. Trash Check
        if pred_idx == 2:
            return None, f"Trash ({confidence:.2f})"
        
        # 2. Confidence Check
        if confidence < CONFIDENCE_THRESHOLD:
            return None, f"Low Confidence ({confidence:.2f})"

        # 3. Valid Return
        if pred_idx == 0:
            return "FLOOR_PLAN", confidence
        elif pred_idx == 1:
            return "HOUSE_EXTERIOR", confidence
            
    except Exception as e:
        print(f"Classification Error: {e}")
        return None, "Error"
    
    return None, "Unknown"

def get_plan_id_from_url(url):
    try:
        slug = url.split('/')[-1]
        return slug.split('-')[-1]
    except:
        return "unknown_id"

def setup_directories():
    os.makedirs(FLOOR_PLAN_DIR, exist_ok=True)
    os.makedirs(HOUSE_IMAGE_DIR, exist_ok=True)

# ==========================================
# 4. CRAWLING LOGIC
# ==========================================

def collect_plan_urls(target_count):
    collected = set()
    page_num = 1
    base = "https://www.architecturaldesigns.com/house-plans/collections/new-arrivals"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    while len(collected) < target_count:
        url = f"{base}?page={page_num}"
        print(f"Scanning Page {page_num}: {url}")
        try:
            r = requests.get(url, headers=headers)
            if r.status_code != 200:
                print(f"Status Code {r.status_code}, stopping.")
                break
            
            soup = BeautifulSoup(r.text, 'html.parser')
            links = soup.find_all('a', href=True)
            new = 0
            for link in links:
                href = link['href']
                if '/house-plans/' in href and '-plan-' in href and 'collections' not in href:
                    full = urljoin(base, href)
                    if full not in collected:
                        collected.add(full)
                        new += 1
                        if len(collected) >= target_count:
                            break
            print(f"  > Found {new} new | Total: {len(collected)}")
            if new == 0:
                break
            page_num += 1
            time.sleep(1)
        except Exception as e:
            print("URL collection error:", e)
            break
    return list(collected)

def crawl_plan_page(url):
    plan_id = get_plan_id_from_url(url)
    print(f"\n--- Processing {plan_id} ---")
    headers = {'User-Agent': 'Mozilla/5.0'}

    try:
        r = requests.get(url, headers=headers)
        if r.status_code != 200: return
        
        soup = BeautifulSoup(r.text, 'html.parser')
        img_tags = soup.find_all('img')

        floor_num = 1
        house_num = 1
        seen_urls = set()

        for img in img_tags:
            src = img.get('src') or img.get('data-src')
            if not src: continue
            
            full_url = urljoin(url, src)
            if full_url in seen_urls: continue
            seen_urls.add(full_url)
            
            # Basic string filtering
            if any(bad in full_url.lower() for bad in ['logo', 'icon', 'avatar', 'svg', 'ads', 'user']):
                continue

            try:
                # Download Image
                img_resp = requests.get(full_url, headers=headers, timeout=10)
                img_data = img_resp.content
                
                # Size check (Ignore tiny thumbnails)
                if len(img_data) < 15000: continue

                # Open PIL Image once
                pil_img = Image.open(BytesIO(img_data))

                # --- 1. ROBUST HASH CHECK ---
                img_hash = compute_dhash(pil_img)
                if is_duplicate(img_hash, seen_image_hashes, threshold=4):
                    # print("  [Skip] Duplicate")
                    continue
                seen_image_hashes.add(img_hash)

                # --- 2. STRICT CLASSIFICATION ---
                category, conf = classify_image_strict(img_data)
                
                if category == "FLOOR_PLAN":
                    print(f"  [SAVE] Floorplan ({conf:.2f})")
                    name = f"{plan_id}_floorplan_{floor_num}.jpg"
                    with open(os.path.join(FLOOR_PLAN_DIR, name), "wb") as f:
                        f.write(img_data)
                    floor_num += 1
                    
                elif category == "HOUSE_EXTERIOR":
                    print(f"  [SAVE] Exterior  ({conf:.2f})")
                    name = f"{plan_id}_house_{house_num}.jpg"
                    with open(os.path.join(HOUSE_IMAGE_DIR, name), "wb") as f:
                        f.write(img_data)
                    house_num += 1
                else:
                    # Trash or Low Confidence
                    # print(f"  [Drop] {conf}") 
                    pass

                time.sleep(0.5) # Be polite

            except Exception as e:
                pass # Skip broken images

    except Exception as e:
        print("Page failed:", e)

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    setup_directories()
    
    # Get URLs
    urls = collect_plan_urls(TARGET_PLAN_COUNT)
    print(f"\nProcessing {len(urls)} URLs...\n")
    
    for i, url in enumerate(urls):
        print(f"[{i+1}/{len(urls)}]")
        crawl_plan_page(url)
        time.sleep(1)
        
    print("DONE!")