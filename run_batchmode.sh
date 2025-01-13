#!/bin/sh
CONFIG=${1:-example_config.toml}
SIM_NAME=pedsilicoICH_$(date +'%m-%d-%Y_%H-%M')
LOG_DIR=logs/$SIM_NAME
SAVE_DIR=/projects01/didsr-aiml/$USER/pedsilicoICH/$SIM_NAME
INPUT=/projects01/didsr-aiml/jayse.weaver/pedsilicoICH/pedsilicoICH_01-10-2025_17-00/pedsilicoICH_01-10-2025_17-00.csv

desired_cases_line=$(grep "desired_cases=" "$CONFIG")
COUNT=$(echo "$desired_cases_line" | sed 's/desired_cases=\([0-9]*\).*/\1/')

echo Running $COUNT simulation conditions

START_TASK=1
END_TASK=$COUNT
qsub -N $SIM_NAME -t $START_TASK-$END_TASK batchmode_CT_dataset_pipeline.sge $LOG_DIR $SAVE_DIR $COUNT $CONFIG $INPUT
