#!/bin/sh

SIM_NAME=pedsilicoICH_$(date +'%m-%d-%Y_%H-%M')
LOG_DIR=logs/$SIM_NAME
SAVE_DIR=/home/jayse.weaver/Datasets/$SIM_NAME
INPUT=/home/jayse.weaver/PedSilicoICH/Tuesday_test.csv

COUNT=10
echo Running $COUNT simulation conditions

START_TASK=1
END_TASK=$COUNT
qsub -N $SIM_NAME -t $START_TASK-$END_TASK batchmode_CT_dataset_pipeline.sge $LOG_DIR $SAVE_DIR $COUNT $INPUT
