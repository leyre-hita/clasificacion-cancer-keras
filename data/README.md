# Dataset
This project uses the Breast Cancer Wisconsin Diagnostic Dataset.

Source: UCI Machine Learning Repository

The dataset contains 569 samples with 30 numerical features describing cell nuclei characteristics such as:
- radius
- texture
- perimeter
- area
- smoothness
- concavity

Target variable:
- 0 → Benign
- 1 → Malignant

The dataset can be loaded directly using scikit-learn:
```python
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
```


