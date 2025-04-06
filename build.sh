#!/bin/bash
# Remove any existing model files to prevent LFS issues
rm -rf model
mkdir -p model

# Download the model directly
wget --no-check-certificate "$MODEL_URL" -O model/brain_tumor_model.h5

# Start the application
python app.py
