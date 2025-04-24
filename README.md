# Drug Response Prediction using Conditional Variational Autoencoder (cVAE)

Predicting therapeutic efficacy from RNA-seq data using deep generative models with biologically relevant context.

## Overview

This project investigates the use of Conditional Variational Autoencoders (cVAEs) to improve drug response prediction in cancer cell lines. By integrating RNA-seq gene expression profiles with additional omics data—such as copy number variations (CNVs), somatic mutations, and hypermethylation levels—the model learns a biologically informed latent representation suitable for both regression (IC50 prediction) and classification (sensitive vs. resistant) tasks.

## Objectives

- Develop a conditional VAE architecture to encode high-dimensional RNA-seq data into a compact latent space.
- Use conditional omics data to guide the model toward more biologically relevant features.
- Train kernel ridge regression models on latent vectors to predict IC50 drug response values.
- Evaluate classification performance using AUROC and AUPR metrics.
- Compare against baseline dimensionality reduction methods (PCA, UMAP) and conventional prediction models.

## Dataset Description

### RNA-seq Data
- Source: Cell Model Passports
- Samples: 969 cancer cell lines
- Features: ~31,000 genes
- Preprocessing:
  - TPM normalization
  - Min-Max scaling for sigmoid compatibility
  - Removal of low-variance genes (variance < 0.1)

### Drug Response Data
- Source: GDSC (Genomics of Drug Sensitivity in Cancer)
- Metric: IC50 values (normalized by max concentration and log-transformed)
- Used for both regression and binary classification tasks

### Conditional Data
- Copy Number Variations (CNVs)
- Somatic Mutations
- Hypermethylation levels
- All conditional data preprocessed and aligned by `model_id` using one-hot encoding and scaling

## Methods

### Model Architecture
- Encoder: Fully connected layers (1024 → 512 → 256 → latent space), ReLU activations
- Decoder: Fully connected layers (latent → 256 → 512 → 1024 → output), ReLU + sigmoid
- Conditional inputs are concatenated with RNA-seq features and latent vectors in encoder/decoder respectively

### Training Strategy
- Loss function: Combination of reconstruction loss (MSE) and KL divergence
- β-VAE approach used to mitigate KL dominance and model collapse (β = 0.01)
- Latent space size tuned; final models trained with size = 20

### Prediction Pipeline
- Latent vectors passed to kernel ridge regression with RBF kernel
- Downstream tasks:
  - Regression: Predict normalized IC50 values
  - Classification: Sensitive vs Resistant via thresholding and SMOTE/3-class stratification

### Validation
- 5-fold cross-validation performed
- Performance measured via:
  - MSE (regression)
  - AUROC, AUPR (classification)

## Results Summary

- The cVAE outperformed baseline methods (standard VAE, PCA, UMAP) in both regression and classification tasks
- Demonstrated robustness in maintaining predictive performance at lower latent space dimensions
- Conditional information significantly improved feature selection when the dimensionality was constrained

## Key Contributions

- Built an end-to-end data science pipeline for bioinformatics applications: data cleaning, normalization, model training, and evaluation
- Integrated domain knowledge (genetic features) into representation learning to improve interpretability and task relevance
- Demonstrated strong potential of cVAEs for personalized medicine and drug response modeling

## Future Work

- Explore alternative architectures (e.g., cross-attention mechanisms, conditional priors)
- Improve preprocessing of conditional inputs to further optimize performance
- Extend to multi-drug or pan-cancer generalization tasks

## Technologies Used

- Python (NumPy, Pandas, Scikit-learn, TensorFlow)
- Matplotlib and Seaborn for visualization
- SMOTE for handling class imbalance
- Jupyter Notebook for experimentation and analysis

## Author

Johan Trippitelli  
McGill University – Class of 2024  
Email: jtrippitelli@gmail.com | LinkedIn: [linkedin.com/in/johan-trippitelli-bb267a220](https://www.linkedin.com/in/johan-trippitelli-bb267a220)

