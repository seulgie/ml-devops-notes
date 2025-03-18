# Data Validation with Kolmogorov-Smirnov Test and Weights & Biases

This document summarizes the process of validating dataset similarity using the Kolmogorov-Smirnov (KS) test while leveraging **Weights & Biases (wandb)** for dataset management.

## 1. Project Setup with `wandb`

Initialize a Weights & Biases (wandb) run to manage dataset artifacts:

```python
import wandb

run = wandb.init(project="exercise_9", job_type="data_tests")
```

## 2. Configuring `pytest` Command-Line Options

In `conftest.py`, custom command-line options are defined to specify reference and sample datasets, along with the KS test's alpha value:

```python
def pytest_addoption(parser):
    parser.addoption("--reference_artifact", action="store")
    parser.addoption("--sample_artifact", action="store")
    parser.addoption("--ks_alpha", action="store")
```

## 3. Loading Data Fixtures

The `data` fixture fetches the reference and sample datasets from `wandb`, ensuring they are available for testing:

```python
@pytest.fixture(scope="session")
def data(request):
    reference_artifact = request.config.option.reference_artifact
    sample_artifact = request.config.option.sample_artifact

    if reference_artifact is None or sample_artifact is None:
        pytest.fail("Missing required artifacts")

    sample1 = pd.read_csv(run.use_artifact(reference_artifact).file())
    sample2 = pd.read_csv(run.use_artifact(sample_artifact).file())

    return sample1, sample2
```

The `ks_alpha` fixture extracts the alpha value for statistical testing:

```python
@pytest.fixture(scope='session')
def ks_alpha(request):
    ks_alpha = request.config.option.ks_alpha
    if ks_alpha is None:
        pytest.fail("--ks_alpha missing on command line")
    return float(ks_alpha)
```

## 4. Kolmogorov-Smirnov Test Implementation

The KS test compares the distributions of key features in both datasets:

```python
import scipy.stats

def test_kolmogorov_smirnov(data, ks_alpha):
    sample1, sample2 = data
    columns = [
        "danceability", "energy", "loudness", "speechiness",
        "acousticness", "instrumentalness", "liveness", "valence",
        "tempo", "duration_ms"
    ]
    
    alpha_prime = 1 - (1 - ks_alpha) ** (1 / len(columns))  # Bonferroni correction
    
    for col in columns:
        ts, p_value = scipy.stats.ks_2samp(sample1[col], sample2[col])
        assert p_value > alpha_prime
```

### Key Points:
- **Bonferroni Correction**: Adjusts the alpha value to account for multiple hypothesis tests.
- **Kolmogorov-Smirnov Test**: Measures the difference between empirical distributions of two samples.
- **Assertion**: If the p-value is too low, the datasets are considered significantly different.

## 5. Running the Tests

Execute the test with:

```sh
pytest test_data.py \
  --reference_artifact "dataset_1.csv:latest" \
  --sample_artifact "dataset_2.csv:latest" \
  --ks_alpha 0.05
```

## Conclusion
This testing framework ensures that dataset distributions remain consistent over time. Using **Weights & Biases** helps track dataset versions, while the **Kolmogorov-Smirnov test** provides statistical validation.

