# mechanistic-interpretability-genz-slang
# MechInterp Gen Z Slang

Discovering and steering a Gen Z slang representation in large language models using **Mechanistic Interpretability**, **PyTorch**, and **TransformerLens**.

## Overview

Large Language Models learn not only factual knowledge but also linguistic styles. This project investigates whether **Gen Z slang** is represented as a distinct direction in the activation space of an LLM and whether that direction can be manipulated to control model outputs.

Using the **Gemma-2-2B-Instruct** model, activation vectors were extracted from multiple transformer layers and analyzed to identify a direction associated with Gen Z slang. Activation steering was then used to amplify or suppress this style during generation.

## Key Features

* Extract hidden activations from Gemma-2-2B
* Identify a Gen Z slang direction in activation space
* Analyze layer-wise style representations
* Visualize activation clustering using PCA
* Inject slang style using activation steering
* Suppress slang through activation ablation
* Perform experiments without modifying model weights

## Methodology

1. Create paired prompts with identical content but different writing styles.
2. Extract residual stream activations using TransformerLens.
3. Compute average activation differences between slang and formal responses.
4. Identify layers with maximum style separation.
5. Visualize activations using Principal Component Analysis (PCA).
6. Apply activation steering to modify generated outputs.
7. Evaluate stylistic changes across multiple steering strengths.

## Results

* Style information forms a measurable representation within the model.
* Activation steering successfully increases slang usage in neutral prompts.
* Activation ablation suppresses slang generation.
* Certain transformer layers encode stylistic information more strongly than others.

## Technologies Used

* Python
* PyTorch
* TransformerLens
* NumPy
* Scikit-Learn
* Plotly
* Hugging Face Transformers
* Gemma-2-2B-Instruct

## Repository Structure

```text
.
├── mech-interp-genz-slang.ipynb
├── REPORT.md
├── README.md
└── assets/
```

## Installation

```bash
pip install torch transformers transformer-lens numpy scikit-learn plotly
```

## Running the Project

Open the notebook and execute all cells:

```bash
jupyter notebook mech-interp-genz-slang.ipynb
```

or use Google Colab.

## Future Work

* Explore additional internet language styles
* Compare style representations across different LLMs
* Investigate multi-dimensional steering vectors
* Extend analysis to sentiment and persona control

## Author

Janhavi
IIT Madras
