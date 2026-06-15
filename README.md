# Visual-Search-Engine-Model-Trained-On-H-M-And-Deepfashion-Dataset
This an offline reverse image search model trained on two incredibly large datasets.(H&amp;M 256x256 || Deepfashion) 


# Offline Visual Search & Ranking Engine

This project implements a comprehensive offline visual search and ranking engine, specifically designed for e-commerce applications, with a focus on categories like modest wear, abaya, and custom garments. The pipeline leverages a hybrid approach combining deep learning embeddings with classical computer vision features, followed by an XGBoost re-ranking mechanism for enhanced relevance.

## Project Overview

The engine processes large image datasets through a multi-phase pipeline to enable efficient and accurate visual similarity search:

1.  **Phase 1: Data Ingestion & Preprocessing**
    -   Handles data ingestion from datasets like H&M 256x256 and DeepFashion2.
    -   Performs necessary preprocessing, including catalog creation and stratified data splitting (80/10/10 for train/validation/test).

2.  **Phase 2: Deep Learning Embeddings**
    -   Fine-tunes an EfficientNet-B3 model with online hard triplet mining.
    -   Generates 512-dimensional L2-normalized deep learning embedding vectors for each image.

3.  **Phase 3: Classical Computer Vision Features**
    -   Extracts 192-dimensional HSV color histograms for robust color matching.
    -   Computes 20-dimensional GLCM texture descriptors to capture fabric patterns.
    -   Concatenates these classical features with the deep learning embeddings to form a rich 724-dimensional hybrid feature vector.

4.  **Phase 4: PCA Compression + FAISS Offline Index**
    -   Applies Principal Component Analysis (PCA) to reduce the 724-dimensional vectors to a compact 128-dimensional representation (retaining ≥95% variance).
    -   Constructs an ultra-fast FAISS IVFFlat index for efficient nearest-neighbor search, enabling sub-millisecond query times entirely offline.

5.  **Phase 5: XGBoost Re-ranking + Query Demo**
    -   Trains an XGBoost Learning to Rank (LTR) model using simulated interaction data (e.g., click-through rates, add-to-cart, purchase rates) to re-score the top-K candidates retrieved by FAISS.
    -   Includes an end-to-end query demonstration, showcasing how the system identifies and ranks visually similar products based on a query image.

## Datasets

-   **H&M 256x256**: Downloaded via KaggleHub (`odins0n/hm256x256`)
-   **DeepFashion2**: Downloaded via KaggleHub (`ashraygupta9/deep-fashion`)

## Setup and Usage

To run this project, clone the repository and follow the instructions in the Jupyter Notebook to install dependencies, process data, train models, and execute queries.
