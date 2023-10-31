#!/bin/bash



# Synchronizing training scripts to Orion
LOCAL_DIR='./training_scripts/'
REMOTE_DIR='dat300-h23-40@filemanager.orion.nmbu.no:~/ca3/training_scripts'
rsync -avzP ${LOCAL_DIR} ${REMOTE_DIR}

