
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass

# Add src to path
sys.path.append(os.path.abspath('src'))

# Import using package structure
from insilicoICH import lesion_definition as ld

# Try to import VITools
try:
    import VITools
except ImportError:
    print("VITools not found. Please ensure it is installed.")
    sys.exit(1)

def main():
    # 1. Load Phantom
    print("Loading phantom...")
    phantom = None
    try:
        phantom_factory = VITools.get_available_phantoms()['6.5 yr NIHPD Head']
        phantom = phantom_factory() # Instantiate
        print("Phantom loaded.")
    except Exception as e:
        print(f"Failed to load phantom: {e}")
        print("Falling back to MOCK phantom.")
        
        # Create Mock Phantom
        # Shape: 100x100x100
        shape = (100, 100, 100)
        img = np.zeros(shape, dtype=np.float32)
        
        # Define structures
        # Brain: Sphere
        z, y, x = np.indices(shape)
        center = np.array(shape) / 2
        r = np.sqrt((x-center[2])**2 + (y-center[1])**2 + (z-center[0])**2)
        brain_mask = r < 40
        
        # Ventricles: Two central ellipsoids
        ventricle_mask = np.zeros(shape, dtype=bool)
        # Left Ventricle
        v1 = ((x-center[2]-10)**2/10 + (y-center[1])**2/40 + (z-center[0])**2/40) < 5
        # Right Ventricle
        v2 = ((x-center[2]+10)**2/10 + (y-center[1])**2/40 + (z-center[0])**2/40) < 5
        ventricle_mask = v1 | v2
        
        # SAH: Peripheral shell (sulci) - random spots on periphery
        sah_mask = (r > 35) & (r < 38) & (np.sin(x/2)*np.sin(y/2) > 0.5)
        
        # Combined CSF
        csf_mask = ventricle_mask | sah_mask
        
        # Setup Mock Object
        class MockPhantom:
            def __init__(self):
                self.shape = shape
                self.spacings = (1.0, 1.0, 1.0)
                self.img = img
            def get_material_mask(self, material):
                if material == 'CSF':
                    return csf_mask
                return np.zeros(shape, dtype=bool)
        
        phantom = MockPhantom()

    # 2. Get CSF Mask
    print("Getting CSF mask...")
    try:
        csf_mask = phantom.get_material_mask('CSF')
        # Ensure it's boolean
        csf_mask = csf_mask > 0.5
        print(f"CSF Mask Shape: {csf_mask.shape}, Volume: {np.sum(csf_mask)} voxels")
    except Exception as e:
        print(f"Failed to get CSF mask: {e}")
        return

    # 3. Partition
    print("Partitioning CSF...")
    ventricles, sah = ld.partition_csf_to_ventricles_and_sah(csf_mask)
    print(f"Ventricle Volume: {np.sum(ventricles)} voxels")
    print(f"SAH Volume: {np.sum(sah)} voxels")

    # Save partition check
    if np.sum(ventricles) > 0:
        # Find a slice with good ventricle representation
        proj = np.sum(ventricles, axis=(1, 2))
        z_slice = np.argmax(proj)
        
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(csf_mask[z_slice], cmap='gray')
        plt.title(f"Original CSF (z={z_slice})")
        plt.subplot(1, 3, 2)
        plt.imshow(ventricles[z_slice], cmap='gray')
        plt.title("Ventricles")
        plt.subplot(1, 3, 3)
        plt.imshow(sah[z_slice], cmap='gray')
        plt.title("SAH")
        plt.savefig('csf_partition.png')
        print("Saved csf_partition.png")
    
    # Check Spacings
    spacings = phantom.spacings if hasattr(phantom, 'spacings') else (1,1,1)
    print(f"Spacings: {spacings}")

    # 4. Generate IVH
    print("Generating IVH...")
    if np.sum(ventricles) > 0:
        ivh_lesion = ld.LesionFactory.create('IVH', boundary=ventricles, spacings=spacings)
        ivh_lesion.generate(volume_ml=5.0)
        print(f"IVH generated. Achieved Volume: {ivh_lesion.volume_ml:.2f} mL")

        if ivh_lesion.mask.any():
            center = center_of_mass(ivh_lesion.mask)
            z_ivh = int(center[0])
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(ventricles[z_ivh], cmap='gray')
            plt.title(f"Ventricle Mask (z={z_ivh})")
            plt.subplot(1, 2, 2)
            plt.imshow(ivh_lesion.mask[z_ivh], cmap='hot')
            plt.title("Generated IVH Mask")
            plt.savefig('ivh_test.png')
            print("Saved ivh_test.png")
    else:
        print("Skipping IVH generation (no ventricles found).")

    # 5. Generate SAH
    print("Generating SAH...")
    if np.sum(sah) > 0:
        sah_lesion = ld.LesionFactory.create('SAH', boundary=sah, spacings=spacings)
        sah_lesion.generate(volume_ml=2.0) 
        print(f"SAH generated. Achieved Volume: {sah_lesion.volume_ml:.2f} mL")

        if sah_lesion.mask.any():
            center_sah = center_of_mass(sah_lesion.mask)
            z_sah = int(center_sah[0])
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(sah[z_sah], cmap='gray')
            plt.title(f"SAH Space Mask (z={z_sah})")
            plt.subplot(1, 2, 2)
            plt.imshow(sah_lesion.mask[z_sah], cmap='hot')
            plt.title("Generated SAH Mask")
            plt.savefig('sah_test.png')
            print("Saved sah_test.png")
    else:
        print("Skipping SAH generation (no SAH space found).")

if __name__ == "__main__":
    main()
