# Test Suite

The test suite validates the core components of the machine-learning course recommender project.

## Coverage

- Dataset validation and summaries
- User-profile content-based recommendations
- Course-similarity recommendations
- Bag-of-Words course similarity
- User-profile feature preparation
- PCA and K-means clustering
- Embedding feature construction
- Regression and classification model training
- Surprise dataset preparation for KNN and NMF
- Neural-network recommender architecture

## Run

From the project root:

```bash
pytest -v
```

The neural-network tests are automatically skipped if TensorFlow is unavailable.
