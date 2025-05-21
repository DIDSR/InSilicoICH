from subprocess import run
from shutil import rmtree
import pandas as pd
from test_image_acquisition import test_dir


def test_cli():
    output_dir = test_dir / "test_cli_output"

    inclusion_criteria = test_dir / 'test_inclusion_criteria.toml'

    run(["recruit", inclusion_criteria, "--output_directory", output_dir])

    input_csv = output_dir / (output_dir.name + '.csv')

    run(["generate", input_csv])

    input_df = pd.read_csv(input_csv)
    output_df = pd.concat([pd.read_csv(o) for o in
                           output_dir.rglob('metadata_*.csv')],
                          ignore_index=True)
    assert len(input_df) == 2
    assert len(output_df) == 20
    rmtree(output_dir)
