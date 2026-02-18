import os
import urllib.request
from pathlib import Path

def download_file(url, output_path):
    """Download file with progress"""
    print(f"Downloading: {output_path.name}")
    urllib.request.urlretrieve(url, output_path)
    print(f"✓ Downloaded: {output_path}")

def setup_plan2scene_models():
    """Download Plan2Scene configs and weights"""
    
    # Create directories
    plan2scene_dir = Path("plan2scene")
    checkpoints_dir = plan2scene_dir / "checkpoints"
    configs_dir = plan2scene_dir / "conf" / "plan2scene"
    
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Downloading Plan2Scene Models (Version 2)")
    print("="*60)
    
    # Texture Synthesis Config
    texture_synth_conf_url = "https://aspis.cmpt.sfu.ca/projects/plan2scene/models/version_2/texture_synth_conf.yml"
    texture_synth_conf_path = configs_dir / "texture_synth_conf.yml"
    
    # Texture Synthesis Weights
    texture_synth_weights_url = "https://aspis.cmpt.sfu.ca/projects/plan2scene/models/version_2/texture_synth_weights.pth"
    texture_synth_weights_path = checkpoints_dir / "texture_synth_weights.pth"
    
    # Texture Propagation Config
    texture_prop_conf_url = "https://aspis.cmpt.sfu.ca/projects/plan2scene/models/version_2/texture_prop_conf.json"
    texture_prop_conf_path = configs_dir / "texture_prop_conf.json"
    
    # Texture Propagation Weights
    texture_prop_weights_url = "https://aspis.cmpt.sfu.ca/projects/plan2scene/models/version_2/texture_prop_weights.pth"
    texture_prop_weights_path = checkpoints_dir / "texture_prop_weights.pth"
    
    # Or use the user's provided checkpoint
    user_checkpoint_url = "https://www.dropbox.com/scl/fi/zo4zwgeyns9nau74xu051/ckpt_epoch200000.pth?rlkey=0hz8l93j1ljuyd30tmmb4p7a5&e=1&dl=1"
    user_checkpoint_path = checkpoints_dir / "ckpt_epoch200000.pth"
    
    files_to_download = [
        (texture_synth_conf_url, texture_synth_conf_path),
        (texture_synth_weights_url, texture_synth_weights_path),
        (texture_prop_conf_url, texture_prop_conf_path),
        (texture_prop_weights_url, texture_prop_weights_path),
        (user_checkpoint_url, user_checkpoint_path),
    ]
    
    for url, path in files_to_download:
        if not path.exists():
            try:
                download_file(url, path)
            except Exception as e:
                print(f"Warning: Could not download {path.name}: {e}")
                print(f"Please manually download from: {url}")
    
    print("\n" + "="*60)
    print("Plan2Scene models setup complete!")
    print("="*60)

def setup_roofgan_models():
    """Download RoofGAN models"""
    
    roofgan_dir = Path("roofgan")
    checkpoints_dir = roofgan_dir / "experiments" / "proj_dir" / "model_gan"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("RoofGAN Models")
    print("="*60)
    
    # RoofGAN pretrained model URL
    roofgan_weights_url = "https://aspis.cmpt.sfu.ca/projects/roofgan/checkpoints.zip"
    
    print(f"Please manually download RoofGAN checkpoints from:")
    print(f"  {roofgan_weights_url}")
    print(f"Extract to: {checkpoints_dir}")

if __name__ == "__main__":
    setup_plan2scene_models()
    setup_roofgan_models()
