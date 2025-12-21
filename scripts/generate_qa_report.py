
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root / 'src'))

from insilicoICH.lesion_definition import LesionFactory

OUTPUT_DIR = project_root / 'qa_output'
OUTPUT_DIR.mkdir(exist_ok=True)

def save_lesion_plot(lesion, boundary_slice, title, filename):
    center_z = lesion.coords_voxel[0]
    if center_z < 0 or center_z >= lesion.mask.shape[0]:
        center_z = lesion.mask.shape[0] // 2
    
    mask_slice = lesion.mask[center_z]
    img_slice = lesion.image[center_z]
    
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(mask_slice, cmap='gray')
    ax[0].set_title(f"{title} Mask (z={center_z})")
    
    ax[1].imshow(img_slice, cmap='gray')
    ax[1].set_title(f"{title} Texture")
    
    if boundary_slice is not None:
        ax[2].imshow(boundary_slice[center_z], cmap='gray')
        ax[2].imshow(mask_slice, cmap='jet', alpha=0.5)
        ax[2].set_title("Overlap")
        
        ax[3].imshow(np.max(lesion.mask, axis=0), cmap='gray')
        ax[3].set_title("MIP (Axial)")
    else:
        ax[2].axis('off')
        ax[3].axis('off')
    
    save_path = OUTPUT_DIR / filename
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")
    print(f"  Target Volume: {lesion.volume_ml:.2f} mL (Achieved)")
    print(f"  Target HU: {lesion.intensity_HU:.2f} (Achieved)")

def run_qa():
    print("Generating QA Report...")

    # 1. EDH
    print("\n--- Generating EDH ---")
    shape = (100, 100, 100)
    dura = np.zeros(shape, dtype=bool)
    z, y, x = np.ogrid[:100, :100, :100]
    center = (50, 50, 50)
    dist_sq = (z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2
    dura[(dist_sq <= 45**2) & (dist_sq >= 44**2)] = True
    
    lesion = LesionFactory.create('EDH', boundary=dura, spacings=(1, 1, 1), seed=1234)
    lesion.generate(volume_ml=10.0, intensity_hu=60)
    save_lesion_plot(lesion, dura, "EDH", "qa_edh.png")

    # 2. IPH
    print("\n--- Generating IPH ---")
    brain = np.zeros(shape, dtype=bool)
    brain[((z-50)**2 + (y-50)**2 + (x-50)**2) <= 40**2] = True
    
    lesion = LesionFactory.create('IPH', boundary=brain, spacings=(1, 1, 1), seed=4321)
    lesion.generate(volume_ml=15.0, intensity_hu=50)
    save_lesion_plot(lesion, brain, "IPH", "qa_iph.png")

    # 3. Fracture
    print("\n--- Generating Fracture ---")
    skull = np.zeros(shape, dtype=np.uint8)
    skull[(dist_sq <= 45**2) & (dist_sq >= 40**2)] = 1
    
    lesion = LesionFactory.create('Fracture', boundary=skull, spacings=(1, 1, 1), seed=5678)
    lesion.generate(fracture_length=50, thickness=2)
    save_lesion_plot(lesion, None, "Fracture", "qa_fracture.png")

if __name__ == "__main__":
    run_qa()
