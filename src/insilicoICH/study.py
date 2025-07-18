"""
pipeline: A refactored high-level module for orchestrating virtual imaging trials
for Intracerebral Hemorrhage (ICH) CT simulations.
"""
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional
from functools import partial

import numpy as np
import pandas as pd
import pydicom
import tomllib
from monai.transforms import RandAffine
from scipy.ndimage import center_of_mass

from VITools import Study, get_available_phantoms, load_vol
from .phantoms.head_phantoms import LesionPhantom
from .lesion_definition import LesionFactory

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
            val_col = f'{lesion_type}_volume' if 'IPH_volume' in df.columns else f'{lesion_type}_HU'
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
        edema: List[int] = [0, 15],
        mass_effect: list[bool | float] = [0.1, 0.9],
        texture_contrast: List[float] = [0, 3],
        texture_scale: List[float] = [8, 16],
        complexity: List[int] = [1, 4],
        smoothness: List[float] = [0.1, 0.4],
        irregularity: List[float] = [0.1, 0.4],
        eccentricity: List[float] = [0.4, 0.8],
        add_augmentation: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generates a DataFrame of study parameters by sampling from distributions.
        """
        phantoms = [(k, v) for k, v in get_available_phantoms().items() if k in phantoms]
        lesion_phantoms = [
            k for k, v in phantoms if
            isinstance(v, partial) and
            issubclass(v.func, LesionPhantom)
            ]
        base_df = super().generate_from_distributions(lesion_phantoms, study_count, **kwargs)
        rng = np.random.default_rng(base_df['global_seed'].iloc[0])

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
            phantom_class = get_available_phantoms()[base_df['phantom'].iloc[i]]
            params = {}

            lesion_type = None
            if hasattr(phantom_class, 'func') and issubclass(phantom_class.func, LesionPhantom):
                lesion_type = rng.choice(subtype)

            params['subtype'] = lesion_type
            params['lesion_volume'] = 0
            params['lesion_attenuation'] = 0
            params['edema'] = 0

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

                params['lesion_volume'] = vol
                params['lesion_attenuation'] = intensity
                if lesion_type == 'IPH':
                    params['edema'] = rng.choice(range(*edema)) # uniform distributions may not be most appropriate here
                    # may need to switch to normal distribution by providing mean/stddev instead of range
                    params['texture_contrast'] = rng.uniform(*texture_contrast)
                    params['texture_scale'] = rng.uniform(*texture_scale)
                    params['complexity'] = rng.integers(*complexity)
                    params['smoothness'] = rng.uniform(*smoothness)
                    params['irregularity'] = rng.uniform(*irregularity)
                    params['eccentricity'] = rng.uniform(*eccentricity)

            age = phantom_class.keywords.get('age', 0)
            params['age'] = age
            mass_effect = [0.4, 0.6] if mass_effect is True else mass_effect
            params['mass_effect'] = rng.uniform(low=min(mass_effect), high=max(mass_effect)) if mass_effect else False
            params['add_augmentation'] = add_augmentation
            study_params.append(params)

        ich_df = pd.DataFrame(study_params)
        return base_df.join(ich_df)

    def load_phantom(self, patient_id: int = 0):
        """Loads the base phantom and inserts a lesion based on study parameters."""
        phantom = super().load_phantom(patient_id)
        series = self.metadata.iloc[patient_id]

        if pd.notna(series.get('subtype')) and series.get('lesion_volume', 0) > 0:
            # This part assumes Lesion objects from your other module are available
            # and can be created via a factory.
            if series.subtype in ['SDH', 'EDH']:
                boundary = phantom.get_dura_map()
            elif series.subtype == 'IPH':
                boundary = phantom.get_material_mask('white matter')
            else:
                boundary = None

            lesion_params = {
                'spacings': phantom.spacings,
                'seed': series.case_seed,
                'boundary': boundary
            }

            lesion_obj = LesionFactory.create(series.subtype, **lesion_params)
            lesion_obj.generate(
                volume_ml=series.lesion_volume,
                intensity_hu=series.lesion_attenuation,
                texture_contrast=getattr(series, 'texture_contrast', 0),
                texture_scale=getattr(series, 'texture_scale', 12),
                complexity=getattr(series, 'complexity', 0),
                smoothness=getattr(series, 'smoothness', 0.2),
                irregularity=getattr(series, 'irregularity', 0.2),
                eccentricity=getattr(series, 'eccentricity', 0.6),
                edema=getattr(series, 'edema', 0),
                # Pass other relevant params from series to generate...
            )

            # This assumes your phantom has an `insert_lesion` method
            # that takes a generated lesion object.
            phantom.insert_lesion(lesion_obj, mass_effect=series.mass_effect)

        # Check for augmentation flag, disable on Windows if needed
        if series.add_augmentation and os.name != 'nt':
            if hasattr(phantom, 'apply_transform'):
                transform = RandAffine(
                    prob=1.0,
                    rotate_range=[np.pi/4, np.pi/20, np.pi/20],
                    translate_range=[10, 10, 10],
                    scale_range=[0.1, 0.1, 0.1],
                    padding_mode="border",
                    mode='nearest'
                )
                phantom.apply_transform(transform, seed=series.case_seed)

        return phantom

    def run_study(self, patient_id: int = 0):
        """Runs the CT simulation and generates post-simulation metadata and masks."""
        results = super().run_study(patient_id)
        series = self.metadata.iloc[patient_id]

        # Initialize default values
        mask_path, lesion_coords, vol_by_slice_ml, slice_intensity = None, None, 0, 0

        if pd.notna(series.subtype):
            # Generate and write lesion mask
            # startZ, endZ = self.scanner.scan_coverage
            mask_vol = self.scanner.get_lesion_mask(
                startZ=self.scanner.scan_coverage[0],
                endZ=self.scanner.scan_coverage[1],
                slice_thickness=series.slice_thickness,
                fov=series.fov
            )

            # --- Create a temporary study object to write the mask ---
            # This avoids modifying the main scanner's recon attribute
            mask_scanner = self.scanner
            mask_scanner.recon = mask_vol
            dicom_path = Path(series.output_directory) / 'lesion_masks'
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
        rows = results.case_id == f'case_{patient_id:04d}'
        results.loc[rows, 'subtype'] = series.subtype
        results.loc[rows, 'lesion_volume(mL)'] = vol_by_slice_ml
        results.loc[rows, 'lesion_attenuation(HU)'] = slice_intensity
        results.loc[rows, 'mass_effect'] = series.mass_effect
        results.loc[rows, 'lesion_location(z,y,x)'] = lesion_coords
        results.loc[rows, 'mask_file_path'] = mask_path

        return results

    def get_masks(self, patientid: int = 0):
        """
        Retrieve the lesion mask volume(s) for a given patient.

        Parameters:
            patientid (int): Index of the patient/case.

        Returns:
            np.ndarray: Loaded mask volume(s).
        """
        return load_vol(self.results[self.results.case_id ==
                                     f'case_{patientid:04d}']['mask_file_path'])

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
        for key in ['lesion_volume', 'lesion_attenuation']:
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
    if 'subtype' in config:
        config['subtype'] = [s or None for s in config['subtype']]

    return config


def recruit_patients_cli(arg_list: Optional[List[str]] = None):
    """CLI for generating a study plan (recruiting patients)."""
    parser = ArgumentParser(description="Generates a study plan CSV from parameter distributions.")
    parser.add_argument('config', nargs='?', help="Path to user-defined TOML config file.")
    # Add all other arguments... (kept brief for example)
    parser.add_argument('--output_directory', '-o', type=str, default='results')
    parser.add_argument('--study_count', type=int)
    args = parser.parse_args(arg_list)

    pkg_dir = Path(__file__).parent
    config = load_and_merge_configs(
        default_config_path=pkg_dir / 'configs/default.toml',
        user_config_path=args.config,
        cli_args=vars(args)
    )

    output_dir = Path(config['output_directory'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter available phantoms by age
    age_range = config.pop('age', (0, 120))
    available_phantoms = get_available_phantoms()
    valid_phantoms = {}
    for k, v in available_phantoms.items():
        if (not isinstance(v, partial)) and (not hasattr(v, 'keywords')):
            continue

        if age_range[0] < v.keywords.get('age', -1) < age_range[1]:
            valid_phantoms[k] = v

    if not valid_phantoms:
        print(f"No phantoms found in age range {age_range}. Exiting.")
        return
    config.pop('config', None)  # Remove config key if present
    df = ICHStudy.generate_from_distributions(list(valid_phantoms.keys()), **config)

    save_name = output_dir / f"{output_dir.name}_study_plan.csv"
    df.to_csv(save_name, index=False)
    print(save_name)


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
