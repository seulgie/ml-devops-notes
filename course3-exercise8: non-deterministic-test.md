# Kolmogorov-Smirnov Data Testing Script

This script performs statistical tests to compare training and test datasets, ensuring they follow similar distributions. It uses the **Kolmogorov-Smirnov (KS) test** to detect significant differences between numerical feature distributions.

## Dependencies
- `pytest`
- `wandb`
- `pandas`
- `scipy.stats`

## Script Overview

The script performs the following steps:
1. Retrieves the latest `train` and `test` datasets from W&B.
2. Extracts key numerical columns.
3. Applies the **KS test** to compare distributions.
4. Uses **Bonferroni correction** to adjust for multiple testing.
5. Fails the test if any feature's distribution significantly differs.

### Main Code

```python
import pytest
import wandb
import pandas as pd
import scipy.stats

# Initialize W&B run for logging
run = wandb.init(project="exercise_8", job_type="data_tests")

@pytest.fixture(scope="session")
def data():
    local_path = run.use_artifact("exercise_6/data_train.csv:latest").file()
    sample1 = pd.read_csv(local_path)

    local_path = run.use_artifact("exercise_6/data_test.csv:latest").file()
    sample2 = pd.read_csv(local_path)

    return sample1, sample2

def test_kolmogorov_smirnov(data):
    sample1, sample2 = data

    numerical_columns = [
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms"
    ]

    alpha = 0.05  # Type I error probability
    alpha_prime = 1 - (1 - alpha) ** (1 / len(numerical_columns))  # Bonferroni correction

    for col in numerical_columns:
        ts, p_value = scipy.stats.ks_2samp(sample1[col], sample2[col], alternative="two-sided")
        assert p_value > alpha_prime  # Test fails if distributions significantly differ
```

## How to Use
RUn the script using `pytest`:
```bash
pytest test_data.py
```

## Key Statistical Concept
- Kolmogorov-Smirnov Test: Compares two distributions and tests if they come from the same underlying population.
- Bonferroni Correction: Adjusts significance thresholds when performing multiple tests to control for false positives.

## Notes
- The script checks only numerical features.
- If test fails, it suggests the `train` and `test` datasets might have different distributions.
- This ensures that training and evaluation datasets are statistically consistent, preventing data leakage or distribution drift.

## Source:
https://github.com/udacity/nd0821-c2-build-model-workflow-exercises/blob/master/lesson-3-data-validation/exercises/exercise_8/solution/test_data.py
