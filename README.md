# Collaborative Code Generation: Artifact Repository

## Description

This repository provides scripts, training datasets, and prompts for reproducing and verifying the results presented in the paper "Promise and Peril of Collaborative Code Generation Models: Balancing Effectiveness and Memorization".

## Table of Contents

- [Installation](#installation)
- [Dataset Construction Guide](#dataset-construction-guide)
- [Model Training Guide](#model-training-guide)
- [Model Evaluation Guide](#model-evaluation-guide)

## Installation

To install the dependencies, simply run the following command:

```bash
pip install -r requirements.txt
```

## Dataset Construction Guide

Instructions on how to construct the dataset for this project.

1. **Collect the Data:**
    - We use the Google BigQuery platform to collect data.
    - The database used is `bigquery-public-data.github_repos`.
    - The query used to collect all Python source codes with licenses from the database is:
    
    ```sql
    SELECT
      f.repo_name, f.path, c.copies, c.size, c.content, l.license
    FROM
      `bigquery-public-data.github_repos.files` AS f
    JOIN
      `bigquery-public-data.github_repos.contents` AS c
    ON
      f.id = c.id
    JOIN
      `bigquery-public-data.github_repos.licenses` AS l
    ON
      f.repo_name = l.repo_name 
    WHERE
      NOT c.binary
        AND ((f.path LIKE '%.py'))
    ```
    
    For more information on Google BigQuery, refer to the [official documentation](https://cloud.google.com/bigquery/docs/introduction).

2. **Preprocess the Data:**
    - Set the path to the dataset and configure in `./scripts/data_reprocessing/argument.py`.
    - Run `python ./scripts/data_reprocessing/extract_orga_repos.py` to extract data for different organizations.
    - Run `python ./scripts/data_reprocessing/preprocess.py` to preprocess datasets.
    - Run `python ./scripts/data_reprocessing/dataset_splitter.py` to split datasets into training and validation datasets.

## Model Training Guide

1. **Train Models:**

    - **Centralized Training:**
        - Set the path to the dataset and configure in `./scripts/model_training/centralized_training/argument.py`.
        - Run `python ./scripts/model_training/centralized_training/centralized_train.py`.

    - **Federated Learning:**
        - Set the path to the dataset and configure in `./scripts/model_training/federated_learning/argument.py`.
        - Start the aggregation server by running:
        ```bash
        python ./scripts/model_training/federated_learning/server_avg.py 
        ```
        or 
        ```bash
        python ./scripts/model_training/federated_learning/server_yogi.py
        ```
        - Then start each client by providing different `node_id` and arguments:
        ```bash
        accelerate launch --gpu_ids='all' client.py --node_id=NODE_ID --local_dataset_path="PATH" --dataset_name_train="TRAINING_DATASET_NAME" --dataset_name_valid="VALIDATION_DATASET_NAME"
        ```

        For more details about the Flower Federated Framework, please refer to https://flower.ai/
    - **Incremental Learning:**
        - Set the path to the dataset and configure in `./scripts/model_training/incremental_learning/argument.py`.
        - Run:
        ```bash
        accelerate launch --gpu_ids='all' ./scripts/model_training/incremental_learning/train.py 
        ```
        each time to train with different datasets in a sequential order.

## Model Evaluation Guide

1. **Evaluate Effectiveness of Models:**
    - **Next Token Prediction:**
        - Run `python ./scripts/effectiveness_evaluation/evaluate_perplexity.py` to get the Perplexity (PPL) score.
    
    - **EvalPlus Benchmark:**
        - Clone the repository:
        ```bash
        git clone https://github.com/bigcode-project/bigcode-evaluation-harness.git
        ```
        - Navigate to the directory:
        ```bash
        cd bigcode-evaluation-harness
        ```
        - Run:
        ```bash
        accelerate launch main.py --model "PATH_TO_MODEL" --tasks humanevalplus --temperature TEMPERATURE --n_samples 200 --batch_size 100 --metric_output_path="PATH_TO_SAVE_RESULT" --allow_code_execution > "PATH_TO_SAVE_LOG" 2>&1
        ```
        For more details about the EvalPlus benchmark, please refer to [EvalPlus GitHub](https://github.com/evalplus/evalplus).

2. **Memorization Evaluation:**
    - **Construct Prompts for Each Training Dataset:**
        - Run `python ./scripts/memorization_evaluation/prompt_construction/extract_prompts.py`.
    - **Code Generation:**
        - Run `python ./scripts/memorization_evaluation/generation/generation.py`.
    - **Memorization Detection:**
        - Run `python ./scripts/memorization_evaluation/memoriztaion_detection/detection.py` to get the detection report.
        - Run `python ./scripts/memorization_evaluation/process_report.py` to extract useful statistics from the report.
