#!/bin/bash
# Remove any existing model files to prevent LFS issues
#rm -rf model
#mkdir -p model

# Download the model directly
#wget --no-check-certificate "https://drive.google.com/file/d/1-3KZAIoDLV98_5f9KH84tL07QyQBawxT/view?usp=sharing" -O model/brain_tumor_model.h5

# Start the application
pip install -r requirements.txt
python -c "from app import download_model; download_model()"
