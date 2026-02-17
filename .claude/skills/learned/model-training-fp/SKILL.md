---
name: model-training-fp
description: Export training data for the false positive (FP) classifier and run the agent workflow for model training.
---

# Model Training Fp Skill

This skill was learned from a workflow recording on 2026-02-17.

## Overview

This workflow demonstrates exporting training data for the FP classifier and attempting to run the model training agent workflow. The user navigates to the project directory, starts a Jupyter server, opens the FP classifier notebook, but the main focus is on preparing to run the agent-based training workflow.

## Steps

### Step 1: Navigate to project directory

Set working directory for the project

**Target**: sportswear-esg-news-classifier project folder

```bash
cd ~/Documents/Courses/DataTalksClub/projects/machine-learning-zoomcamp/sportswear-esg-news-classifier
```

**Notes**: User navigated through multiple directory levels to reach the project root

<details>
<summary>Supporting evidence from recording</summary>

- fdpearce@workhorse22: ~/Documents/Courses/DataTalksClub/projects/machine-learning-zoomcamp/sportswear-esg-news-classifier

</details>

### Step 2: Start Jupyter Lab server

Launch notebook interface for potential model development

**Target**: Jupyter Lab environment

```bash
uv run --with jupyter jupyter lab
```

**Notes**: Used uv run to ensure proper virtual environment activation

<details>
<summary>Supporting evidence from recording</summary>

- [I 2026-02-17 00:53:08.970 ServerApp] Jupyter Server 2.17.6 is running at:
- http://localhost:8888/lab

</details>

### Step 3: Open FP classifier EDA notebook

Review the false positive classifier notebook for feature engineering

**Target**: fp1_EDA_FE.ipynb notebook

**Notes**: Opened the first notebook in the FP classifier workflow which handles exploratory data analysis and feature engineering

<details>
<summary>Supporting evidence from recording</summary>

- False Positive Brand Classifier - EDA & Feature Engineering
- Target Brands (50)
- Nike Adidas Puma Under Armour Lululemon

</details>

### Step 4: Export training data for FP classifier

Prepare training data as mentioned in the audio narration

**Target**: FP classifier training dataset

```bash
uv run python scripts/export_training_data.py --dataset fp
```

**Notes**: Audio indicates this is the intended action, though the actual command execution is not shown in the screen content

<details>
<summary>Supporting evidence from recording</summary>

- test is to run the model training agent workflow using the following command. Let's take off the agent and we'll export the training data for the FP classifier

</details>

### Step 5: Run model training agent workflow

Execute the automated model training pipeline

**Target**: Agent orchestrator model training workflow

```bash
uv run python -m src.agent run model_training
```

**Notes**: This was the main objective mentioned in the audio, though the execution is not completed in this recording

<details>
<summary>Supporting evidence from recording</summary>

- test is to run the model training agent workflow using the following command

</details>

## Metadata

- **Recorded**: 2026-02-17 08:50 UTC
- **Workflow**: test-run
- **Description**: quick trial
- **Recording Duration**: 3 minutes
