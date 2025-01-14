#!/bin/sh
INPUT=${1:-input.csv}
SIM_NAME=pedsilicoICH_$(date +'%m-%d-%Y_%H-%M')
LOG_DIR=logs/$SIM_NAME
SAVE_DIR=/projects01/didsr-aiml/$USER/pedsilicoICH/$SIM_NAME

COUNT=$(cat $INPUT | wc -l)
COUNT=$(($COUNT - 1))
echo Running $COUNT simulation conditions

START_TASK=1
END_TASK=$COUNT
qsub -N $SIM_NAME -t $START_TASK-$END_TASK batchmode_CT_dataset_pipeline.sge $LOG_DIR $SAVE_DIR $COUNT $INPUT
