#!/bin/bash
TRAINING_DATA_PATH='gs://jk-demo-datasets/gasdata/training.csv'
TESTING_DATA_PATH='gs://jk-demo-datasets/gasdata/testing.csv'
REGION="us-central1"
ARTIFACT_BUCKET='gs://jk-demo-artifacts'
JOBDIR_BUCKET='gs://jk-demo-jobdir'

JOB_NAME="JOB_TEST_11"
JOB_DIR="${JOBDIR_BUCKET}/${JOB_NAME}"


SCALE_TIER="BASIC"
MODULE_NAME="trainer.train"
RUNTIME_VERSION="1.14"
PYTHON_VERSION="3.5"

N_FEATURES_OPTIONS="[2,4,6]"
L2_REG_OPTIONS="[0.1,0.2,0.3,0.5]"

TRAINING_APP_FOLDER='training_app/trainer'

gcloud ai-platform jobs submit training $JOB_NAME \
--region $REGION \
--job-dir $JOB_DIR \
--package-path $TRAINING_APP_FOLDER \
--module-name $MODULE_NAME \
--scale-tier $SCALE_TIER \
--python-version $PYTHON_VERSION \
--runtime-version $RUNTIME_VERSION \
-- \
--data_path $TRAINING_DATA_PATH \
--n_features_options $N_FEATURES_OPTIONS \
--l2_reg_options $L2_REG_OPTIONS
