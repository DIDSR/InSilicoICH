"""
pipeline: A refactored high-level module for orchestrating virtual imaging trials
for Intracerebral Hemorrhage (ICH) CT simulations.
"""
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pydicom
import tomllib
from monai.transforms import RandAffine
from scipy.ndimage import center_of_mass

from VITools import Study, get_available_phantoms
from .phantoms.head_phantoms import LesionPhantom

# --- Constants and Configuration ---
LESION_TYPES = list(LesionPhantom.lesion_types)

# =============================================================================
# MODULE 1: CONFIGURATION AND DATA MANAGEMENT
# Helper class to manage loading and sampling from parameter distributions.
# This encapsulates the repetitive logic from the original script.
# =============================================================================


class DistributionManager:
    """Handles loading, parsing, and sampling from parameter distributions."""

    def __init__(self, source: str | Path | Dict[str, List[float]]):
        self.distributions = {}
        if isinstance(source, (str, Path)):
            self._load_from_csv(Path(source))
        elif isinstance(source, dict):
            self._load_from_dict(source)
        else:
            raise TypeError(f"Unsupported source type for DistributionManager: {type(source)}")

    def _load_from_csv(self, path: Path):
        """Loads distributions from a CSV file with value and weight columns."""
        df = pd.read_csv(path)
        for lesion_type in LESION_TYPES:
            val_col = f'{lesion_type}_volume' if 'volume' in df.columns[0] else f'{lesion_type}_HU'
            weight_col = f'{lesion_type}_weight'
            if val_col in df.columns and weight_col in df.columns:
                subset_df = df[[val_col, weight_col]].dropna()
                weights = subset_df[weight_col] / subset_df[weight_col].sum()
                self.distributions[lesion_type] = {
                    'values': subset_df[val_col].values,
                    'weights': weights.values
                }

    def _load_from_dict(self, config_dict: Dict[str, List[float]]):
        """Creates uniform distributions from a dictionary of min/max ranges."""
        for lesion_type, min_max in config_dict.items():
            values = np.linspace(min_max[0], min_max[1], num=100)
            self.distributions[lesion_type] = {
                'values': values,
                'weights': np.full_like(values, 1/len(values))
            }

    def sample(self, lesion_type: str, rng: np.random.Generator) -> float:
        """Samples a single value for a given lesion type."""
        dist = self.distributions.get(lesion_type)
        if dist is None:
            raise ValueError(f"No distribution found for lesion type: {lesion_type}")
        return rng.choice(dist['values'], p=dist['weights'])

# =============================================================================
# MODULE 2: CORE STUDY ORCHESTRATION
# The main ICHStudy class, now cleaner and more focused.
# =============================================================================


class ICHStudy(Study):
    """
    Manages the generation and execution of in silico ICH virtual trials.
    """
    @classmethod
    def generate_from_distributions(
        cls,
        phantoms: List[str],
        study_count: int = 1,
        subtype: List[Optional[str]] = [None] + LESION_TYPES,
        lesion_volume: Dict | str | Path = None,
        lesion_attenuation: Dict | str | Path = None,
        edema_range: List[int] = [0, 15],
        mass_effect: bool = True,
        add_augmentation: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generates a DataFrame of study parameters by sampling from distributions.
        """
        base_df = super().generate_from_distributions(phantoms, study_count, **kwargs)
        rng = np.random.default_rng(base_df['GlobalSeed'].iloc[0])

        # Use the DistributionManager for clean handling of inputs
        vol_manager = DistributionManager(lesion_volume)
        att_manager = DistributionManager(lesion_attenuation)

        # Define lesion-specific sampling rules in a config dictionary
        lesion_rules = {
            'EDH': {'vol_limit': float('inf'), 'att_limit': 45},
            'SDH': {'vol_limit': float('inf'), 'att_limit': 45},
            'IPH': {'vol_limit': 50, 'att_limit': 45}
        }

        study_params = []
        for i in range(study_count):
            phantom_class = get_available_phantoms()[base_df['Phantom'].iloc[i]]
            params = {}
            
            lesion_type = None
            if hasattr(phantom_class, 'func') and issubclass(phantom_class.func, LesionPhantom):
                lesion_type = rng.choice(subtype)
            
            params['Subtype'] = lesion_type
            params['LesionVolume'] = 0
            params['LesionAttenuation'] = 0
            params['Edema'] = 0

            if lesion_type:
                rules = lesion_rules[lesion_type]
                # Improved sampling loop to avoid potential infinite loops
                vol, intensity = 0, 0
                for _ in range(100): # Max 100 retries
                    vol = vol_manager.sample(lesion_type, rng)
                    if vol <= rules['vol_limit']: break
                for _ in range(100):
                    intensity = att_manager.sample(lesion_type, rng)
                    if intensity >= rules['att_limit']: break
                
                params['LesionVolume'] = vol
                params['LesionAttenuation'] = intensity
                if lesion_type == 'IPH':
                    params['Edema'] = rng.choice(range(*edema_range))

            age = phantom_class.keywords.get('age', 0)
            params['Age'] = age
            params['MassEffect'] = mass_effect
            params['AddAugmentation'] = add_augmentation
            study_params.append(params)

        ich_df = pd.DataFrame(study_params)
        return base_df.join(ich_df)

    def load_phantom(self, patient_id: int = 0):
        """Loads the base phantom and inserts a lesion based on study parameters."""
        series = self.metadata.iloc[patient_id]
        phantom = super().load_phantom(patient_id)

        if pd.notna(series.Subtype) and series.LesionVolume > 0 and hasattr(phantom, 'insert_lesion'):
            phantom.insert_lesion(
                subtype=series.Subtype,
                volume=series.LesionVolume,
                intensity=series.LesionAttenuation,
                mass_effect=series.MassEffect,
                seed=series.CaseSeed,
                edema=int(series.Edema)
            )
        
        # Check for augmentation flag, disable on Windows if needed
        if series.AddAugmentation and os.name != 'nt':
            if hasattr(phantom, 'apply_transform'):
                transform = RandAffine(
                    prob=1.0,
                    rotate_range=[np.pi/4, np.pi/20, np.pi/20],
                    translate_range=[10, 10, 10],
                    scale_range=[0.1, 0.1, 0.1],
                    padding_mode="border",
                    mode='nearest'
                )
                phantom.apply_transform(transform, seed=series.CaseSeed)
        
        return phantom

    def run_study(self, patient_id: int = 0):
        """Runs the CT simulation and generates post-simulation metadata and masks."""
        results = super().run_study(patient_id)
        series = self.metadata.iloc[patient_id]
        
        # Initialize default values
        mask_path, lesion_coords, vol_by_slice_ml, slice_intensity = None, None, 0, 0

        if pd.notna(series.Subtype):
            # Generate and write lesion mask
            mask_vol = self.scanner.get_lesion_mask(
                startZ=self.scanner.ScanCoverage[0],
                endZ=self.scanner.ScanCoverage[1],
                slice_thickness=series.SliceThickness,
                fov=series.FOV
            )
            
            # --- Create a temporary study object to write the mask ---
            # This avoids modifying the main scanner's recon attribute
            mask_scanner = self.scanner.__class__(self.scanner.phantom)
            mask_scanner.recon = mask_vol
            dicom_path = Path(series.OutputDirectory) / 'lesion_masks'
            patient_name = self.scanner.phantom.patient_name
            mask_path = mask_scanner.write_to_dicom(dicom_path / f'{patient_name}_mask.dcm')
            
            # Calculate metrics from the mask
            dcm = pydicom.dcmread(mask_path[0])
            spacings = [float(dcm.SliceThickness)] + list(map(float, dcm.PixelSpacing))
            voxel_vol_ml = np.prod(spacings) / 1000.0
            
            vol_by_slice_ml = mask_vol.sum(axis=(1, 2)) * voxel_vol_ml
            
            z, y, x = center_of_mass(mask_vol) # Note: order is z,y,x for numpy
            lesion_coords = f"[{z:.1f}, {y:.1f}, {x:.1f}]"
            
            slice_intensity = np.zeros_like(vol_by_slice_ml)
            if hasattr(self.scanner.phantom, 'lesion_intensity'):
                slice_intensity[vol_by_slice_ml > 0] = self.scanner.phantom.lesion_intensity

        # Update results DataFrame
        rows = results.CaseID == f'case_{patient_id:04d}'
        results.loc[rows, 'Subtype'] = series.Subtype
        results.loc[rows, 'LesionVolume(mL)'] = vol_by_slice_ml
        results.loc[rows, 'LesionAttenuation(HU)'] = slice_intensity
        results.loc[rows, 'MassEffect'] = series.MassEffect
        results.loc[rows, 'LesionLocation(z,y,x)'] = lesion_coords
        results.loc[rows, 'MaskFilePath'] = mask_path
        
        return results

# =============================================================================
# MODULE 3: COMMAND-LINE INTERFACE (CLI)
# Refactored CLI logic for clarity and separation of concerns.
# =============================================================================

def _flatten_dict(layered_dict: Dict) -> Dict:
    """Helper to flatten nested dictionaries from TOML files."""
    config = {}
    for v in layered_dict.values():
        if isinstance(v, dict):
            config.update(v)
    return config

def load_and_merge_configs(
    default_config_path: Path,
    user_config_path: Optional[str],
    cli_args: Dict
) -> Dict:
    """Loads and merges configurations from default, user, and CLI sources."""
    # 1. Load default config
    with open(default_config_path, 'rb') as f:
        config = tomllib.load(f)
        config = _flatten_dict(config)
        # Resolve relative paths for distributions
        for key in ['LesionVolume', 'LesionAttenuation']:
            if key in config:
                config[key] = default_config_path.parent / config[key]

    # 2. Override with user config file
    if user_config_path:
        with open(user_config_path, 'rb') as f:
            user_config = tomllib.load(f)
            config.update(_flatten_dict(user_config))
    
    # 3. Override with CLI arguments (only non-None values)
    cli_config = {k: v for k, v in cli_args.items() if v is not None}
    config.update(cli_config)

    # Final cleanup
    if 'Subtype' in config:
        config['Subtype'] = [s or None for s in config['Subtype']]
        
    return config

def recruit_patients_cli(arg_list: Optional[List[str]] = None):
    """CLI for generating a study plan (recruiting patients)."""
    parser = ArgumentParser(description="Generates a study plan CSV from parameter distributions.")
    parser.add_argument('config', nargs='?', help="Path to user-defined TOML config file.")
    # Add all other arguments... (kept brief for example)
    parser.add_argument('--OutputDirectory', '-o', type=str, default='results')
    parser.add_argument('--StudyCount', type=int)
    args = parser.parse_args(arg_list)

    pkg_dir = Path(__file__).parent
    config = load_and_merge_configs(
        default_config_path=pkg_dir / 'configs/default.toml',
        user_config_path=args.config,
        cli_args=vars(args)
    )

    output_dir = Path(config['OutputDirectory'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter available phantoms by age
    age_range = config.pop('Age', (0, 120))
    available_phantoms = get_available_phantoms()
    valid_phantoms = {
        k: v for k, v in available_phantoms.items()
        if age_range[0] < v.keywords.get('age', -1) < age_range[1]
    }

    if not valid_phantoms:
        print(f"No phantoms found in age range {age_range}. Exiting.")
        return

    df = ICHStudy.generate_from_distributions(list(valid_phantoms.keys()), **config)
    
    save_name = output_dir / f"{output_dir.name}_study_plan.csv"
    df.to_csv(save_name, index=False)
    print(f"Study plan with {len(df)} cases saved to: {save_name}")

def run_simulation_cli(arg_list: Optional[List[str]] = None):
    """CLI for running simulations from a study plan CSV."""
    parser = ArgumentParser(description="Runs InSilicoICH simulations from a study plan.")
    parser.add_argument('input_csv', nargs='?', help="Path to study plan CSV file.")
    parser.add_argument('--parallel', '-p', action='store_true', help="Run simulations in parallel.")
    args = parser.parse_args(arg_list)

    input_csv_path = args.input_csv
    if not input_csv_path and not sys.stdin.isatty():
        input_csv_path = sys.stdin.read().strip()
    
    if not input_csv_path:
        parser.error("An input CSV file is required either as an argument or via stdin.")
        
    print(f"Running study from: {input_csv_path}")
    ICHStudy(input_csv_path).run_all(parallel=args.parallel)

if __name__ == '__main__':
    # A simple router to decide which CLI to run
    if len(sys.argv) > 1 and sys.argv[1] == 'recruit':
        recruit_patients_cli(sys.argv[2:])
    else:
        run_simulation_cli()