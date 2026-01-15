#!/bin/sh
#SBATCH --time=700:00:00        # time limit
if [ $(generate --help | grep InSilicoICH | wc -l) -eq 1 ]; then
    generate results/results_study_plan.csv --parallel --chunk_size 2000 #--overwrite
else
    echo "Error: Invalid version of generate function present. Please follow the steps below to reinstall InSilicoICH:"
    echo "1. pip uninstall InSilicoICH -y"
    echo "2. pip install --upgrade git+https://github.com/DIDSR/InSilicoICH.git"
fi
