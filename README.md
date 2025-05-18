# Math Formula Recognizer
A CNN based system for recognizing handwritten mathematical formulas and converting them to LaTeX or ASCII.


## Project Overview
This project implements an end-to-end pipeline that processes images of handwritten mathematical formulas and converts them into machine-readable formats. Using a combination of computer vision techniques and deep learning, the system:
1. Preprocesses input images to isolate individual mathematical symbols
2. Classifies each symbol using a custom-trained CNN
3. Analyzes spatial relationships to detect exponents, fractions, and other mathematical structures
4. Reconstructs the complete formula in LaTeX or ASCII format


## Architecture & Pipeline
Input Image(image path / decoded base64) → `preprocess.py` → Symbol Crops → `inference.py` → Symbol Predictions → `postprocess.py` → Spatial Relationships → `combination.py` → LaTeX/ASCII


## Getting Started
### Clone the repository
git clone https://github.com/imxiaoxiaohh/project-showcase.git

### Install dependencies
pip install -r requirements.txt

### Run the API server
python src/api.py

## Model Training
The CNN was trained interactively in Google Colab (T4 GPU). All the code lives in `trainCNN.ipynb`, which covers:
1. **Data Preparation**  
   - Load the handwritten symbol dataset  
   - Split into training & validation sets  
2. **CNN Architecture**  
   - 5-layer convolutional network with ReLU activations and BatchNorm  
3. **Training Loop**  
   - Optimizer: Adam  
   - Tracks training/validation loss & accuracy over epochs  
   - Saves the best model weights to `models/math_cnn.pth`
