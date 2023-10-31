#!/bin/bash

# Synchronizing model-files from Orion
REMOTE_DIR='dat300-h23-40@filemanager.orion.nmbu.no:~/ca3/models/'
LOCAL_DIR='./models'
rsync -avzP ${REMOTE_DIR} ${LOCAL_DIR}

# Synchronizing output logs from Orion
REMOTE_DIR='dat300-h23-40@filemanager.orion.nmbu.no:~/ca3/training_scripts/output_logs/'
LOCAL_DIR='./training_scripts/outout_logs/'
rsync -avzP ${REMOTE_DIR} ${LOCAL_DIR}

