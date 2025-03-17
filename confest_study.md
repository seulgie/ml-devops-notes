# **Using `conftest.py` in Pytest: A Complete Guide**

## **1. What is `conftest.py`?**
`conftest.py` is a special configuration file in **pytest** used for:
- Defining **fixtures** that can be shared across multiple test files.
- Adding **custom command-line options**.
- Configuring **pytest hooks**.
- Avoiding **code duplication** in test files.

Unlike regular test files (`test_*.py`), pytest **automatically loads** `conftest.py`, so there’s no need to import it manually.

---

## **2. Common Uses of `conftest.py`**
### **(1) Defining Fixtures**
Fixtures provide reusable data and configurations for tests.
```python
import pytest

@pytest.fixture
def sample_data():
    return {"name": "Alice", "age": 25}
```
✅ **Usage in `test_*.py`:**
```python
def test_user(sample_data):
    assert sample_data["age"] == 25
```

### **(2) Adding Command-Line Options**
```python
def pytest_addoption(parser):
    parser.addoption("--env", action="store", default="dev", help="Set environment: dev/staging/prod")
```
✅ **Usage in tests:**
```python
@pytest.fixture
def environment(request):
    return request.config.getoption("--env")

def test_env(environment):
    assert environment in ["dev", "staging", "prod"]
```
✅ **Run with:**
```bash
pytest test_env.py --env=staging
```

### **(3) Using Pytest Hooks**
Hooks modify pytest behavior before/after execution.
```python
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")
```
✅ **Usage in `test_*.py`:**
```python
import pytest

@pytest.mark.slow
def test_heavy_computation():
    assert 2 + 2 == 4
```
✅ **Exclude slow tests:**
```bash
pytest -m "not slow"
```

---

## **3. Example: `conftest.py` for Data Testing**
### **(1) `conftest.py` (Data Fixtures)**
```python
import pytest
import pandas as pd
import wandb

run = wandb.init(project="exercise_9", job_type="data_tests")

def pytest_addoption(parser):
    parser.addoption("--reference_artifact", action="store")
    parser.addoption("--sample_artifact", action="store")
    parser.addoption("--ks_alpha", action="store")

@pytest.fixture(scope="session")
def data(request):
    reference_artifact = request.config.option.reference_artifact
    sample_artifact = request.config.option.sample_artifact
    
    local_path = run.use_artifact(reference_artifact).file()
    sample1 = pd.read_csv(local_path)
    
    local_path = run.use_artifact(sample_artifact).file()
    sample2 = pd.read_csv(local_path)
    
    return sample1, sample2

@pytest.fixture(scope='session')
def ks_alpha(request):
    return float(request.config.option.ks_alpha)
```

### **(2) `test_data.py` (Kolmogorov-Smirnov Test)**
```python
import scipy.stats

def test_kolmogorov_smirnov(data, ks_alpha):
    sample1, sample2 = data

    columns = [
        "danceability", "energy", "loudness", "speechiness", "acousticness",
        "instrumentalness", "liveness", "valence", "tempo", "duration_ms"
    ]

    alpha_prime = 1 - (1 - ks_alpha)**(1 / len(columns))
    
    for col in columns:
        ts, p_value = scipy.stats.ks_2samp(sample1[col], sample2[col])
        assert p_value > alpha_prime
```
✅ **Run with:**
```bash
pytest test_data.py \  
  --reference_artifact "dataset_1.csv:latest" \  
  --sample_artifact "dataset_2.csv:latest" \  
  --ks_alpha 0.05
```

---

## **4. Key Differences Between `conftest.py` and Test Files**
| Feature | `conftest.py` | `test_*.py` |
|---------|--------------|-------------|
| Purpose | Configurations, Fixtures, Hooks | Actual test functions |
| Auto-loaded? | ✅ Yes | ❌ No (must be imported) |
| Defines Fixtures? | ✅ Yes | ✅ Yes |
| Adds CLI Options? | ✅ Yes | ❌ No |

---

## **5. Why Use `conftest.py`?**
✅ **Avoids code duplication** across multiple test files.  
✅ **Enhances test maintainability** (modify once, affect all tests).  
✅ **Allows flexible test configuration** via command-line options.  
✅ **Improves reusability** by defining shared fixtures and hooks.  

By utilizing `conftest.py`, we ensure a **clean, maintainable, and efficient testing workflow** in pytest. 🚀

