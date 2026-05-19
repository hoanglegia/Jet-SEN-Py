# Test Workflow: Fingerprint Recognition System
## 1. Purpose
The goal of this testing workflow is to validate each component of the TinyML pipeline:
- **Data Integrity:** Ensuring labels are correctly extracted from file paths.
- **Preprocessing:** Verifying image resizing and normalization.
- **Model Architecture:** Validating the Siamese Network structure.
- **Quantization Logic:** Ensuring the Symmetric Min-Max conversion works as expected.
- **Export Pipeline:** Verifying that the TFLite model transforms correctly into a C-header array.

## 2. Prerequisites
Testing is performed using ``pytest``. Ensure the following dependencies are installed:
```Bash
pip install pytest tensorflow opencv-python numpy
```

## 3. Test Structure
The tests are located in the ``/tests`` directory and categorized by module:
| Test File | Description |
|-----------|-------------|
| ``test_features.py`` | Validates label extraction from SOCOFing filenames and image resizing logic.|
| ``test_data.py`` | Tests the ``DataGenerator`` to ensure it correctly pairs matched/unmatched fingerprints and normalizes pixel values.|
| ``test_models.py`` | Verifies the Siamese CNN architecture (shared weights) and the INT8 quantization mathematical logic.|
| ``test_export.py`` | Confirms that the binary conversion from TFLite to C-compatible arrays maintains data integrity.|

## 4. How to Run Tests
To execute the test suite, run the following command from the root directory:
```Bash
# Standard execution
python -m pytest tests/

# For a detailed trace in case of failure
python -m pytest tests/ --full-trace
```
## 5. Recent Fixes & Optimizations
During the development of this workflow, the following critical issues were identified and resolved:
- **Infinite Loop Prevention:** Adjusted the ``DataGenerator`` logic to handle cases where the label dictionary contains insufficient entries, preventing the generator from hanging indefinitely during unmatch-pair selection.
- **Key Coverage:** Ensured ``label_real_dict`` contains all possible subject keys used in training batches to avoid ``KeyError``.

## 6. Latest Test Result
**Status:** ``PASSED``
**Total Tests:** 7
**Execution Time:** ~8.83 seconds (on Windows platform)

```text
tests\test_data.py ..                                    [ 28%]
tests\test_export.py .                                   [ 42%]
tests\test_features.py ..                                [ 71%] 
tests\test_models.py ..                                  [100%]
======================= 7 passed in 8.83s =======================
```

## 7. Future Integration
This test suite is designed to be integrated into a **GitHub Actions** or **GitLab CI** pipeline. On every ``push`` or ``pull_request``, these tests will automatically run to prevent regression and ensure that changes to the feature engineering logic do not break the embedded deployment header generation.

