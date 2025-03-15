# Dataset Splitting Script

This script splits a dataset into training and test sets, uploads the resulting splits as W&B artifacts, and logs them. The script uses `scikit-learn`'s `train_test_split` function for splitting and supports optional stratified splitting.

## Dependencies
- `pandas`
- `wandb`
- `scikit-learn`

## Script Overview

The script performs the following steps:
1. Downloads an input dataset artifact from W&B.
2. Splits the dataset into training and test sets.
3. Saves the splits as temporary CSV files.
4. Uploads the splits as W&B artifacts for tracking.

### Main Code

```python
def go(args):
    run = wandb.init(project="exercise_6", job_type="split_data")

    # Download and read the dataset artifact
    logger.info("Downloading and reading artifact")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    df = pd.read_csv(artifact_path, low_memory=False)

    # Split data into train and test sets
    logger.info("Splitting data into train and test")
    splits = {}

    splits["train"], splits["test"] = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df[args.stratify] if args.stratify != 'null' else None,
    )

    # Save splits as temporary files
    with tempfile.TemporaryDirectory() as tmp_dir:
        for split, df in splits.items():
            artifact_name = f"{args.artifact_root}_{split}.csv"
            temp_path = os.path.join(tmp_dir, artifact_name)

            # Save and upload split dataset to W&B
            logger.info(f"Uploading the {split} dataset to {artifact_name}")
            df.to_csv(temp_path)

            artifact = wandb.Artifact(
                name=artifact_name,
                type=args.artifact_type,
                description=f"{split} split of dataset {args.input_artifact}",
            )
            artifact.add_file(temp_path)

            logger.info("Logging artifact")
            run.log_artifact(artifact)
            artifact.wait()
```

### How to Use (Example Command)
```python
python split_data.py \
  --input_artifact "my_dataset:v1" \
  --artifact_root "my_dataset" \
  --artifact_type "dataset_split" \
  --test_size 0.2 \
  --random_state 42 \
  --stratify "label"
```

Source: https://github.com/udacity/nd0821-c2-build-model-workflow-exercises/blob/master/lesson-2-data-exploration-and-preparation/exercises/exercise_6/solution/run.py
