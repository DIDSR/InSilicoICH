from shutil import rmtree
from pathlib import Path

import pandas as pd

from insilicoICH.study import run_simulation_cli, recruit_patients_cli

test_dir = Path(__file__).parent.absolute()


def test_cli():
    output_dir = test_dir / "test_cli_output"

    inclusion_criteria = test_dir / 'test_inclusion_criteria.toml'

    recruit_patients_cli([str(inclusion_criteria), "--output_directory", str(output_dir)])

    input_csv = output_dir / (output_dir.name + '.csv')

    run_simulation_cli([str(input_csv)])

    input_df = pd.read_csv(input_csv)
    output_df = pd.concat([pd.read_csv(o) for o in
                           output_dir.rglob('metadata_*.csv')],
                          ignore_index=True)
    assert len(input_df) == 2
    assert len(output_df) == 20
    rmtree(output_dir)
