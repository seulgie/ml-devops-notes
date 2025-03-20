# MLOps Project Overview

## Table of Contents
1. Introduction
2. Project Structure
3. `main.py` - Configuration and Execution
4. `run.py` - Model Training and Evaluation
5. Key Features and Enhancements

## 1. Introduction
This project implements an MLOps pipeline using **Hydra, MLflow, and Weights & Biases (WandB)**. The main objective is to train and evaluate a **Random Forest** model for genre classification using structured and textual data. The pipeline automates **data loading, model training, logging, and evaluation**.

## 2. Project Structure
```
mlops_project/
├── main.py                 # Entry point script
├── random_forest/
│   ├── run.py              # Training script
│   ├── random_forest_config.json # Model configuration file (generated dynamically)
├── config.yaml             # Configuration file
```

## 3. `main.py` - Configuration and Execution
This script loads the configuration using **Hydra**, sets up **WandB** for logging, and runs `run.py` using **MLflow**.

### Key Functions

#### 3.1 Load Configuration with Hydra
```python
@hydra.main(config_path='.', config_name='config')
def go(config: DictConfig):
```
- Loads `config.yaml`, which includes parameters for **data, model, and experiment tracking**.
- Example `config.yaml`:
```yaml
main:
  project_name: "mlops_project"
  experiment_name: "random_forest_experiment"

data:
  train_data: "dataset.csv"

random_forest:
  n_estimators: 100
  max_depth: 10
  random_state: 42
```

#### 3.2 Set Up WandB Environment
```python
os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]
```
- Ensures that all runs are grouped under the same **WandB experiment**.

#### 3.3 Save Model Configuration
```python
model_config = os.path.abspath("random_forest_config.json")
with open(model_config, "w+") as fp:
    json.dump(dict(config["random_forest"]), fp)
```
- Converts the **Random Forest hyperparameters** into a JSON file.

#### 3.4 Run MLflow Job
```python
_ = mlflow.run(
    os.path.join(root_path, "random_forest"),
    "main",
    parameters={
        "train_data": config["data"]["train_data"],
        "model_config": model_config
    },
)
```
- Calls `run.py` using MLflow with **data and model configuration parameters**.

## 4. `run.py` - Model Training and Evaluation
This script **loads the dataset, trains a Random Forest model, evaluates performance, and logs results to WandB**.

### Key Functions

#### 4.1 Parse Command-Line Arguments
```python
parser = argparse.ArgumentParser(description="Train a Random Forest")
parser.add_argument("--train_data", type=str, required=True)
parser.add_argument("--model_config", type=str, required=True)
args = parser.parse_args()
```
- Accepts dataset and model configuration file paths as inputs.

#### 4.2 Load Data and Preprocess
```python
df = pd.read_csv(args.train_data, low_memory=False)
X = df.copy()
y = X.pop("genre")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
```
- Loads CSV data and splits it into **training (70%) and validation (30%) sets**.

#### 4.3 Build Preprocessing Pipeline
```python
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OrdinalEncoder(), categorical_features),
        ("nlp", TfidfVectorizer(binary=True), nlp_features),
    ],
    remainder="drop",
)
```
- Prepares pipelines for **numerical, categorical, and textual features**.

#### 4.4 Train Random Forest Model
```python
pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(**model_config))
])
pipe.fit(X_train, y_train)
```
- Integrates preprocessing and **Random Forest model into a single pipeline**.

#### 4.5 Evaluate Model Performance
```python
score = roc_auc_score(y_val, pipe.predict_proba(X_val), average="macro", multi_class="ovo")
wandb.log({"AUC": score})
```
- Computes **ROC-AUC score** and logs it to WandB.

#### 4.6 Log Feature Importance and Confusion Matrix
```python
fig_feat_imp, sub_feat_imp = plt.subplots(figsize=(10, 10))
feat_imp = pipe["classifier"].feature_importances_
sub_feat_imp.bar(range(len(feat_imp)), feat_imp, color="r", align="center")
wandb.log({"feature_importance": wandb.Image(fig_feat_imp)})
```
- Extracts and logs **feature importance** as a bar chart.
```python
fig_cm, sub_cm = plt.subplots(figsize=(10, 10))
y_pred = pipe.predict(X_val)
cm = confusion_matrix(y_true=y_val, y_pred=y_pred, normalize="true")
ConfusionMatrixDisplay(cm).plot(ax=sub_cm)
wandb.log({"confusion_matrix": wandb.Image(fig_cm)})
```
- Logs the **confusion matrix** visualization.

## 5. Key Features and Enhancements

### Current Features
✅ **End-to-end MLOps pipeline** using Hydra, MLflow, and WandB  
✅ **Random Forest training with structured and text data**  
✅ **Feature importance and confusion matrix visualization**  
✅ **Automatic hyperparameter configuration** from `config.yaml`  

---
This project effectively demonstrates an **MLOps workflow for model training, evaluation, and logging**. It can be further extended with additional **model tracking, deployment, and automation** features.

