import os
import sys
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from datasets.adapter import CarinthiaAdapter, MIICAdapter, NISTAdapter

def main():
    print("Compiling dataset inventory metadata...")
    
    # 1. Carinthia
    carinthia = CarinthiaAdapter()
    c_obs = carinthia[0]
    carinthia_meta = {
        "dataset": "Carinthia SEM Defect Dataset",
        "number_of_observations": len(carinthia),
        "image_dimensions": f"{c_obs.degraded_image.shape[1] * 4}x{c_obs.degraded_image.shape[0] * 4} (HR), {c_obs.degraded_image.shape[1]}x{c_obs.degraded_image.shape[0]} (LR)",
        "image_type": str(c_obs.degraded_image.dtype),
        "paired_unpaired": "Paired (degraded generated from HR source)",
        "ground_truth_available": "Yes (synthetic HR ground truth)",
        "degradation_type": "Synthetic (controlled blur, Poisson shot noise, downsampling)",
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "source": "Zenodo (https://zenodo.org/records/10715190)"
    }
    
    # 2. MIIC
    miic = MIICAdapter()
    m_obs = miic[0]
    miic_meta = {
        "dataset": "MIIC - Microscopic Images of Integrated Circuits",
        "number_of_observations": len(miic),
        "image_dimensions": f"{m_obs.degraded_image.shape[1] * 4}x{m_obs.degraded_image.shape[0] * 4} (HR), {m_obs.degraded_image.shape[1]}x{m_obs.degraded_image.shape[0]} (LR)",
        "image_type": str(m_obs.degraded_image.dtype),
        "paired_unpaired": "Paired (degraded generated from HR source)",
        "ground_truth_available": "Yes (synthetic HR ground truth)",
        "degradation_type": "Synthetic (controlled blur, Poisson shot noise, downsampling)",
        "license": "Academic Non-Commercial (https://github.com/wenbihan/MIIC-IAD)",
        "source": "GitHub (https://github.com/wenbihan/MIIC-IAD)"
    }
    
    # 3. NIST (Paired Mode)
    nist_paired = NISTAdapter(set_num=1, paired=True)
    np_obs = nist_paired[0]
    nist_paired_meta = {
        "dataset": "NIST Detection Limits for SEM (Paired)",
        "number_of_observations": len(nist_paired),
        "image_dimensions": f"{np_obs.degraded_image.shape[1]}x{np_obs.degraded_image.shape[0]} (LR), {np_obs.ground_truth.shape[1]}x{np_obs.ground_truth.shape[0]} (HR)",
        "image_type": str(np_obs.degraded_image.dtype),
        "paired_unpaired": "Paired (degraded matched to noise-free reference)",
        "ground_truth_available": "Yes (noise-free reference image set*_cex_noise_000_contrast_100.tiff)",
        "degradation_type": "Real Controlled (ARTIMAGEN simulated Poisson noise/contrast grid)",
        "license": "Public Domain (U.S. Government Work)",
        "source": "NIST PDR (https://doi.org/10.18434/mds2-3838)"
    }
    
    # 4. NIST (Blind Mode)
    nist_blind = NISTAdapter(set_num=1, paired=False)
    nb_obs = nist_blind[0]
    nist_blind_meta = {
        "dataset": "NIST Detection Limits for SEM (Blind)",
        "number_of_observations": len(nist_blind),
        "image_dimensions": f"{nb_obs.degraded_image.shape[1]}x{nb_obs.degraded_image.shape[0]}",
        "image_type": str(nb_obs.degraded_image.dtype),
        "paired_unpaired": "Unpaired",
        "ground_truth_available": "No",
        "degradation_type": "Real Controlled (ARTIMAGEN simulated Poisson noise/contrast grid)",
        "license": "Public Domain (U.S. Government Work)",
        "source": "NIST PDR (https://doi.org/10.18434/mds2-3838)"
    }
    
    inventory = {
        "Carinthia": carinthia_meta,
        "MIIC": miic_meta,
        "NIST_Paired": nist_paired_meta,
        "NIST_Blind": nist_blind_meta
    }
    
    inventory_path = "experiments/dataset_inventory.json"
    os.makedirs("experiments", exist_ok=True)
    with open(inventory_path, "w") as f:
        json.dump(inventory, f, indent=4)
        
    print(f"Dataset inventory metadata written to {inventory_path}")

if __name__ == "__main__":
    main()
