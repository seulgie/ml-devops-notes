# ML DevOps - Exercise 3 Summary

## Overview
This exercise processes the **Iris dataset** using **t-SNE visualization** and manages artifacts using **Weights & Biases (W&B)** and **MLflow**. The main steps include:

1. **Downloading Data** (from W&B artifact)
2. **Processing Data** (t-SNE transformation)
3. **Uploading Processed Data** (to W&B as an artifact)
4. **Managing Workflow Execution** (via `main.py` and MLflow)

---

## Key Components

### **1. Data Processing (`process_data/run.py`)**
- **Downloads Iris dataset** from W&B.
- **Applies t-SNE transformation** to reduce dimensions for visualization.
- **Generates a KDE plot** using Seaborn to visualize clusters.
- **Uploads transformed dataset** and visualization to W&B.

#### **Main Steps:**
```python
# Download dataset from W&B
artifact = run.use_artifact(args.input_artifact)
iris = pd.read_csv(artifact.file())

# Convert target values to class names
target_names = ["setosa", "versicolor", "virginica"]
iris["target"] = [target_names[k] for k in iris["target"]]

# Apply t-SNE transformation
tsne = TSNE(n_components=2, init="pca", random_state=0)
transf = tsne.fit_transform(iris.iloc[:, :4])
iris["tsne_1"], iris["tsne_2"] = transf[:, 0], transf[:, 1]

# Generate visualization
sns.displot(iris, x="tsne_1", y="tsne_2", hue="target", kind="kde")

# Upload processed data to W&B
iris.to_csv("clean_data.csv")
run.log_artifact(artifact)
```

### **2. Workflow Management (`main.py`)**
- Uses **Hydra** for configuration management.
- Executes data processing steps using **MLflow**:
  - **Step 1:** Download raw data.
  - **Step 2:** Process the data (t-SNE transformation).
- Configures **W&B project and experiment names** dynamically.

#### **Main Steps:**
```python
# Set up W&B experiment
os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

# Execute MLflow pipeline
mlflow.run(os.path.join(root_path, "download_data"), "main", parameters={
    "file_url": config["data"]["file_url"],
    "artifact_name": "iris.csv",
    "artifact_type": "raw_data",
    "artifact_description": "Input data"
})

mlflow.run(os.path.join(root_path, "process_data"), "main", parameters={
    "input_artifact": "iris.csv:latest",
    "artifact_name": "clean_data.csv",
    "artifact_type": "processed_data",
    "artifact_description": "Cleaned data"
})
```

---

## Tools & Technologies Used
- **Weights & Biases (W&B)** → Artifact logging & visualization.
- **MLflow** → Workflow execution & experiment tracking.
- **Hydra** → Configuration management.
- **t-SNE (sklearn)** → Dimensionality reduction.
- **Seaborn & Matplotlib** → Data visualization.

---

## Next Steps
- Extend processing pipeline with additional transformations.
- Automate deployment & monitoring.
- Explore hyperparameter tuning for better visualization.

