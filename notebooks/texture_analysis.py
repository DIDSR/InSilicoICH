import os
from pathlib import Path
from typing import List, Dict, Any, Type

from dotenv import load_dotenv
import pandas as pd

from measures import (HssayeniLoader, BHSDLoader, InstanceLoader,
                      DatasetLoader, run_feature_extraction_pipeline)


def process_dataset(config: Dict[str, Any]) -> pd.DataFrame | None:
    """
    Processes a single dataset based on a configuration dictionary.

    Args:
        config (Dict[str, Any]): A dictionary containing the dataset's
                                 environment variable key, loader class, and name.

    Returns:
        pd.DataFrame | None: A DataFrame with the extracted features, or None if
                             the dataset path is not found.
    """
    print(f"--- Starting processing for {config['name']} dataset ---")

    # 1. Check for environment variable and path existence
    directory_path_str = os.environ.get(config['env_var'])
    if not directory_path_str:
        print(f"Warning: Environment variable '{config['env_var']}' not set. Skipping dataset.")
        return None

    directory_path = Path(directory_path_str)
    if not directory_path.exists():
        print(f"Warning: Directory not found at '{directory_path}'. Skipping dataset.")
        return None

    print(f"Found dataset at: {directory_path}")

    # 2. Instantiate loader and run the pipeline
    loader_class: Type[DatasetLoader] = config['loader']
    loader = loader_class(dataset_path=directory_path)
    montage_file = None
    if config.get('save_montages', False):
        montage_file = config['save_path'] / f"{config['name']}_montages.png"
        print(f"Montages will be saved to: {montage_file}")
    features_df, _, _ = run_feature_extraction_pipeline(loader,
                                                        filename=montage_file,
                                                        montage_max=config.get('montage_max', 100))

    print(f"Successfully extracted features for {len(features_df)} lesions from {config['name']}.")
    return features_df


def main(save_path: str = '.', save_montage: bool = True, montage_max=100) -> None:
    """
    Main function to orchestrate the feature extraction from multiple datasets,
    combine them, and save the final result.
    """
    # Load environment variables from a .env file
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    load_dotenv()

    # --- Configuration for all datasets to be processed ---
    # To add a new dataset, simply add a new dictionary to this list.
    dataset_configs = [
        {
            "name": "Hssayeni",
            "env_var": "HSSAYENI_DIRECTORY",
            "loader": HssayeniLoader,
        },
        {
            "name": "BHSD",
            "env_var": "BHSD_DIRECTORY",
            "loader": BHSDLoader,
        },
        {
            "name": "INSTANCE",
            "env_var": "INSTANCE_DIRECTORY",
            "loader": InstanceLoader,
        },
        # Example of how to add another dataset in the future:
        # {
        #     "name": "NewDataset",
        #     "env_var": "NEWDATASET_DIRECTORY",
        #     "loader": NewDatasetLoader,
        # }
    ]

    all_dataframes: List[pd.DataFrame] = []

    # Loop through the configuration and process each dataset
    for config in dataset_configs:
        config['save_path'] = save_path
        config['save_montages'] = save_montage
        config['montage_max'] = montage_max
        df = process_dataset(config)
        if df is not None and not df.empty:
            all_dataframes.append(df)

    print("\n--- Combining all datasets ---")

    # Check if any dataframes were successfully processed
    if not all_dataframes:
        print("No datasets were processed. Exiting without creating a combined file.")
        return

    # Combine all dataframes into a single one
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    # Define the output file path
    output_filename = save_path / "combined_radiomics_features.csv"

    # Save the final combined dataframe to a single CSV file
    try:
        combined_df.to_csv(output_filename, index=False)
        print(f"\nSuccessfully saved combined data for {len(combined_df)} lesions to '{output_filename}'.")
    except IOError as e:
        print(f"Error: Could not write to file '{output_filename}'. Reason: {e}")


if __name__ == "__main__":
    main()
