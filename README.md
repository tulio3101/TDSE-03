# TDSE-03 — Convolutional Layers: Architecture, Bias, and Experiments

## Project Title
Exploring Convolutional Layers Through Data and Experiments

## Project Description
This project investigates how convolutional layers introduce inductive bias in neural networks. Instead of treating CNNs as black boxes, the work designs, analyzes, and tests a purposeful convolutional architecture on a real, image-based dataset. The notebook includes exploratory data analysis (EDA), a non-convolutional baseline, a custom CNN design, and controlled experiments to study a single architectural factor (e.g., kernel size or depth). The goal is to connect architectural decisions with learning behavior and interpretability.

## Learning Objectives
- Understand the role and mathematical intuition behind convolutional layers.
- Analyze how architectural decisions (kernel size, depth, stride, padding) affect learning.
- Compare convolutional layers with fully connected layers for image-like data.
- Perform a minimal but meaningful exploratory data analysis (EDA) for NN tasks.
- Communicate architectural and experimental decisions clearly.

## Dataset
**Selected dataset:** _To be filled by the student._

**Why this dataset is appropriate for CNNs**
Provide a short justification focusing on:
- Image-based structure (2D/3D tensors)
- Multiple classes
- Fits in memory for local training

**Dataset summary (EDA checklist)**
- Dataset size and class distribution
- Image dimensions and channels
- Examples of samples per class
- Required preprocessing (normalization, resizing, etc.)

## Repository Contents
- **Notebook:** The full analysis, models, experiments, and results.
- **README:** This document with the project overview and results summary.

## Methods

### 1) Baseline Model (Non-Convolutional)
**Architecture:** Flatten + Dense layers (no convolution).

**Report in the notebook:**
- Architecture diagram / summary
- Number of parameters
- Training and validation performance
- Observed limitations

### 2) Convolutional Architecture (Custom Design)
**Design goals:** Simple but intentional.

**Explicit choices to justify:**
- Number of convolutional layers
- Kernel sizes
- Stride and padding
- Activation functions
- Pooling strategy (if any)

### 3) Controlled Experiments (Single Factor)
Select one factor and keep everything else fixed.

Examples:
- Kernel size: 3×3 vs 5×5
- Number of filters
- Depth: 1 vs 2 vs 3 conv layers
- With vs without pooling
- Effect of stride on feature maps

**Report:**
- Quantitative results (accuracy, loss)
- Qualitative observations
- Trade-offs (performance vs complexity)

## Interpretation and Architectural Reasoning
Answer clearly in your own words:
- Why did convolutional layers outperform (or not) the baseline?
- What inductive bias does convolution introduce?
- In what problems would convolution not be appropriate?

## SageMaker Deployment
This project requires:
- Training the model on SageMaker
- Deploying the model to a SageMaker endpoint

Include the deployment steps and endpoint testing in the notebook.

## Results Summary
Summarize key outcomes here after completing experiments:
- Baseline vs CNN accuracy
- Best CNN configuration
- Key architectural insights

## Getting Started

### Prerequisites
- Python 3.10+ (or the version required by your notebook)
- Jupyter Notebook / JupyterLab
- Deep learning framework (TensorFlow or PyTorch)
- AWS CLI configured (for SageMaker)

### Installation
1. Create and activate a virtual environment.
2. Install dependencies used in the notebook.
3. Launch Jupyter and open the notebook.

## Running the Notebook
1. Open the notebook in Jupyter.
2. Run all cells from top to bottom.
3. Ensure results (tables/plots) are saved in the notebook.

## Testing
If you add tests, describe how to run them here.

## Deployment
Describe how the SageMaker training job and endpoint deployment are executed (links to notebook sections are acceptable).

## Architecture Diagrams
Add a simple diagram for:
- Baseline network
- CNN architecture

## Built With
- Jupyter Notebook
- Python
- TensorFlow or PyTorch
- AWS SageMaker

## Authors
- Tulio

## License
This project is licensed under the MIT License. See LICENSE for details.

## Acknowledgments

