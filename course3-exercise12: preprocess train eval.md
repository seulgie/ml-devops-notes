# Code Breakdown for `exercise12.py`

This script is designed to train a **Random Forest Classifier** using **MLflow**, **W&B (Weights & Biases)**, and **Scikit-learn**, while also supporting model export.

---

## 1. Imports and Logging Setup
```python
import argparse
import logging
import os

import yaml
import tempfile
import mlflow
import pandas as pd
import numpy as np
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, FunctionTransformer
import matplotlib.pyplot as plt
import wandb
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
```
- Loads required libraries for **MLflow**, **data preprocessing**, **training**, and **evaluation**.
- Sets up logging to track events during execution.

```python
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()
```
- Configures logging to show timestamps and messages.

---

## 2. Main Training Function (`go(args)`)
```python
def go(args):
    run = wandb.init(job_type="train")
```
- Initializes a **W&B run** to track training metrics.

```python
    logger.info("Downloading and reading test artifact")
    train_data_path = run.use_artifact(args.train_data).file()
    df = pd.read_csv(train_data_path, low_memory=False)
```
- Downloads and reads the training dataset.

```python
    logger.info("Extracting target from dataframe")
    X = df.copy()
    y = X.pop("genre")
```
- Separates **features (`X`)** from the **target variable (`y`)**.

```python
    logger.info("Splitting train/val")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
```
- Splits the dataset into **training (70%)** and **validation (30%)**.

```python
    logger.info("Setting up pipeline")
    pipe = get_training_inference_pipeline(args)
```
- Calls `get_training_inference_pipeline(args)`, which **builds the preprocessing & training pipeline**.

```python
    logger.info("Fitting")
    pipe.fit(X_train, y_train)
```
- **Trains the pipeline** on the training dataset.

```python
    pred = pipe.predict(X_val)
    pred_proba = pipe.predict_proba(X_val)
```
- Generates **predictions** (`pred`) and **probability scores** (`pred_proba`).

```python
    logger.info("Scoring")
    score = roc_auc_score(y_val, pred_proba, average="macro", multi_class="ovo")
    run.summary["AUC"] = score
```
- Computes **AUC score** and logs it to **W&B**.

---

## 3. Exporting the Model
```python
    if args.export_artifact != "null":
        export_model(run, pipe, X_val, pred, args.export_artifact)
```
- If the user specifies an export filename, **saves the trained model**.

---

## 4. Generating Feature Importance Plot
```python
    fig_feat_imp = plot_feature_importance(pipe)
```
- Calls `plot_feature_importance()` to **visualize important features**.

```python
    fig_cm, sub_cm = plt.subplots(figsize=(10, 10))
```
- Prepares a figure for the **confusion matrix**.

```python
    y_pred = pipe.predict(X_val)
    cm = confusion_matrix(
                y_true=y_val,
                y_pred=y_pred,
                labels=pipe["classifier"].classes_,
                normalize="true"
            )
```
- Generates a **normalized confusion matrix**.

```python
    disp  = ConfusionMatrixDisplay(
                    confusion_matrix=cm,
                    display_labels=pipe["classifier"].classes_
                )
    disp.plot(
        ax=sub_cm,
        values_format=".1f",
        xticks_rotation=90,
    )
```
- Plots the confusion matrix.

```python
    fig_cm.tight_layout()
```
- Adjusts figure layout.

```python
    run.log(
        {
            "feature_importance": wandb.Image(fig_feat_imp),
            "confusion_matrix": wandb.Image(fig_cm),
        }
    )
```
- **Logs plots** to W&B.

---

## 5. Model Export Function (`export_model()`)
```python
def export_model(run, pipe, X_val, val_pred, export_artifact):
```
- Saves the trained model for later use.

```python
    signature = infer_signature(X_val.to_numpy(), val_pred)
```
- **Infers model signature** (input/output types).

```python
    with tempfile.TemporaryDirectory() as temp_dir:
        export_path = os.path.join(temp_dir, "model_export")
```
- Creates a **temporary directory** to store the exported model.

```python
        mlflow.sklearn.save_model(
            pipe,
            export_path,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            signature=signature,
            input_example=X_val.iloc[:2],
        )
```
- Saves the **pipeline as an MLflow model**.

```python
        artifact = wandb.Artifact(
            export_artifact,
            type="model_export",
            description="Random Forest pipeline export",
        )
        artifact.add_dir(export_path)
        run.log_artifact(artifact)
        artifact.wait()
```
- Logs the model as a **W&B artifact**.

---

### Summary
- **Trains a Random Forest classifier**
- **Logs metrics and plots to W&B**
- **Exports model using MLflow**
- **Supports categorical, numerical, and NLP features**

Let me know if you need additional details!

