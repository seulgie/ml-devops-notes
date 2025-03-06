## 📝 Exercise 2: Download Data & Upload to W&B Artifact  

### Overview  
This exercise demonstrates how to download a file from a URL, create a W&B artifact, and upload the file to W&B for tracking and versioning. The process is automated using a Python script with a specified Conda environment and a project configuration.

### Key Components

1. **MLproject File**  
   The `MLproject` file specifies the project name, Conda environment, and entry points. The `download_data` entry point takes four parameters:  
   - `file_url`: URL of the file to download  
   - `artifact_name`: Name of the W&B artifact  
   - `artifact_type`: Type of the artifact (default: `raw_data`)  
   - `artifact_description`: Description for the artifact  

2. **Conda Environment (`conda.yml`)**  
   The `conda.yml` file defines the necessary dependencies:  
   - `requests`: For downloading the file  
   - `wandb`: For creating and uploading artifacts  

3. **Python Script (`download_data.py`)**  
   The script handles the downloading of the file and uploading it to W&B as an artifact:  
   - **Argument Parsing**: Takes command-line arguments for file URL, artifact name, type, and description  
   - **File Download**: Downloads the file using the `requests` library, with streaming support to handle large files  
   - **Artifact Creation**: Uploads the downloaded file to W&B as an artifact, with metadata containing the original file URL  

### Execution Flow
1. The Python script is executed with command-line arguments specifying the URL, artifact name, type, and description.
2. The file is downloaded in chunks and saved in a temporary file.
3. A W&B artifact is created and the file is uploaded.
4. The artifact is logged and tracked using W&B.

### Example Command
```bash
python download_data.py --file_url <file_url> \
                        --artifact_name <artifact_name> \
                        --artifact_type <artifact_type> \
                        --artifact_description <artifact_description>
