# HCML-NLP-Project
**Human Centered Machine Learning (INFOMHCML) Final Project**

## Project Overview

This project focuses on explainable AI techniques for drug review sentiment analysis, comparing various machine learning models and their interpretability methods. We evaluate models using multiple explainability metrics including complexity, faithfulness, sparsity, robustness, and max-sensitivity.

## Models Evaluated

- **LightGBM** (Full Features & Top Features)
- **Lasso Regression** (Full Features & Top Features) 
- **BERT** (Fine-tuned for sentiment classification)
- **ProtoLM** (Prototype-based interpretable language model)

## Explainability Methods

- **SHAP** (SHapley Additive exPlanations)
- **LIME** (Local Interpretable Model-agnostic Explanations)
- **Integrated Gradients** (for BERT)
- **Built-in Prototypes** (for ProtoLM)

## Project Structure

```
HCML-NLP-Project/
├── README.md                           # This file
├── .gitignore                         # Git ignore rules
├── dataset.PNG                        # Dataset visualization
├── feature extraction pipeline.png    # Feature extraction workflow
├── shap_force_plot*.html             # SHAP visualization outputs
│
├── Data/                              # Dataset files
│
├── Models/                            # Trained model artifacts
│
├── src/                               # Source code
│   ├── evaluate.py                    # Model evaluation and metrics
│   ├── feature_extraction.py         # Feature extraction utilities
│   ├── check_the_data.py             # Data validation
│   ├── bert.py                       # BERT model implementation
│   │
│   └── proto_lm/                     # ProtoLM implementation
│       ├── ProtoLM.py                # Main ProtoLM model
│       ├── proto_data_class.py       # Data handling classes
│       ├── drugs_reviews_proto_*.ipynb  # Training notebooks
│       ├── tb_logs*/                 # TensorBoard logs
│       └── readme.md                 # ProtoLM documentation
│
├── Evaluation/                        # Evaluation results and metrics
├── Output/                           # Generated outputs
├── Project Proposal/                 # Initial project proposal
└── Report/                           # Final project report
```

## Key Features

### 1. Multi-Model Comparison
- Traditional ML models (LightGBM, Lasso)
- Deep learning models (BERT)
- Interpretable-by-design models (ProtoLM)

### 2. Comprehensive Evaluation Metrics
- **Complexity**: Model explanation complexity
- **Faithfulness**: How well explanations reflect model behavior
- **Sparsity**: Proportion of important features
- **Robustness**: Stability of explanations (Local Lipschitz Estimate)
- **Max-Sensitivity**: Sensitivity to input perturbations
- **MAE**: Mean Absolute Error for prediction accuracy

### 3. Advanced Visualization
-  SHAP and LIME instanses plots
-  Model comparison charts
- Optuna training plots

## Installation & Setup

```bash
# Clone the repository
git clone <repository-url>
cd HCML-NLP-Project

# Option 1: Using conda (recommended)
conda env create -f environment.yml
conda activate hcml-nlp

# Option 2: Using pip
pip install -r requirements.txt

```

## Model Performance Summary

![Model Performance Comparison](src/proto_lm/model_comparison_normalized.png)


## Key Findings

This project examines the potential of interpretable methods in an NLP task and how they impact model accuracy. We approached this by predicting drug review ratings using four models: Lasso Regression, LightGBM, BERT, and ProtoLM. Each model was assessed for both predictive performance and interpretability using the Quantus evaluation framework. While BERT achieved the highest accuracy, it lacked interpretability. Contrary to expectations, Lasso fell short on interpretability. LightGBM performed moderately on both interpretability and accuracy, while ProtoLM performed poorly overall. Our findings highlight the difficulty of building models that are both accurate and interpretable in NLP tasks and underscore the need for better inherently interpretable approaches.


## License

This project is part of the Human Centered Machine Learning course at Utrecht University.


## Acknowledgments

- Utrecht University HCML Course
- ProtoLM implementation based on [original paper/repository]
- Quantus library for explainability metrics

---