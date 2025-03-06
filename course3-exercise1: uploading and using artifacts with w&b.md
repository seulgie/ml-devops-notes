# Course 3 - Exercise 1: Uploading and Using Artifacts with W&B

## 1. **Uploading an Artifact to W&B** (`upload_artifact.py`)

This script uploads a file as an artifact to [Weights & Biases](https://wandb.ai). It takes the following parameters:  
- `--input_file`: The file to upload.  
- `--artifact_name`: The name for the artifact.  
- `--artifact_type`: The type of the artifact (e.g., dataset, model).  
- `--artifact_description`: A description of the artifact.

### Key Steps:
1. **Initialize W&B Run**: Creates a run in the project "exercise_1" with the job type `upload_file`.
2. **Create Artifact**: Defines the artifact with the given name, type, and description.
3. **Upload File**: Adds the specified file to the artifact and logs it to the W&B run.

---

## 2. **Using an Artifact from W&B** (`use_artifact.py`)

This script retrieves and reads the content of an artifact from W&B. It requires:  
- `--artifact_name`: The name and version of the artifact to fetch.

### Key Steps:
1. **Initialize W&B Run**: Creates a run in the "exercise_1" project with the job type `use_file`.
2. **Retrieve Artifact**: Fetches the artifact using its name.
3. **Read Artifact Content**: Opens the file stored in the artifact and prints its content.

---
