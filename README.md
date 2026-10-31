# Machine Learning Course Recommender System

## Overview

This project develops and evaluates a machine learning-based course recommendation system using user profiles, course metadata, course descriptions, and historical course ratings.

The project implements multiple recommendation approaches covered throughout the machine learning course, including:

- Content-based recommendation
- Course similarity recommendation
- User-profile clustering
- KNN collaborative filtering
- Non-negative Matrix Factorization (NMF)
- Neural-network collaborative filtering
- Embedding-based regression
- Embedding-based classification
- Model evaluation and comparison

The objective is to investigate how different recommendation techniques perform on the same course-rating dataset and identify effective approaches for personalized course recommendation.

---

## Project Objectives

The main objectives are to:

1. Explore and understand the course recommendation dataset.
2. Analyze user profiles, course genres, and course ratings.
3. Develop content-based recommendation methods.
4. Generate recommendations using course-to-course similarity.
5. Segment users using clustering techniques.
6. Implement collaborative filtering using KNN.
7. Implement matrix factorization using NMF.
8. Develop a neural-network recommendation model using user and item embeddings.
9. Use learned embeddings for regression and classification tasks.
10. Compare model performance using appropriate evaluation metrics.
11. Identify the strongest-performing approaches for the dataset.

---

## Dataset

The project uses the IBM Machine Learning course recommendation dataset.

### Raw datasets

The following datasets are stored in `data/raw/`:

| Dataset | Description |
|---|---|
| `course_genre.csv` | Course IDs, titles, and genre/category features |
| `course_processed.csv` | Course information including titles and descriptions |
| `ratings.csv` | Historical user-course ratings |
| `user_profile.csv` | User profile and interest features |

### Dataset dimensions

| Dataset | Rows | Columns |
|---|---:|---:|
| Course genres | 307 | 16 |
| Courses | 307 | 3 |
| Ratings | 233,306 | 3 |
| User profiles | 33,901 | 15 |

The datasets contained no missing values or duplicate rows during exploratory analysis.

A notable finding was that only **126 of the 307 courses** appeared in the ratings dataset. This means approximately 41% of the catalog had observed rating interactions.

The ratings dataset contains ratings of **2 and 3**, with a strong concentration around rating 3.

---

## Project Workflow

```text
                    Course Recommendation System
                              │
             ┌────────────────┴────────────────┐
             │                                 │
      Exploratory Data Analysis        Feature Engineering
             │                                 │
             ├──────────────┬──────────────────┤
             │              │                  │
      User Profiles     Course Text       Course Genres
             │              │                  │
             ▼              ▼                  ▼
       Clustering       BoW / Similarity   Profile Matching
             │              │                  │
             └──────────────┴──────────────────┘
                            │
                            ▼
                 Content-Based Methods
                            │
                            ▼
              Collaborative Filtering
                 │          │          │
                 ▼          ▼          ▼
                KNN        NMF     Neural Network
                                      │
                                      ▼
                                User/Item Embeddings
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
                    Regression               Classification
                         │                         │
                         └────────────┬────────────┘
                                      ▼
                              Model Evaluation
```

---

# Project Structure

```text
ml-course-recommender/
│
├── data/
│   ├── raw/
│   │   ├── course_genre.csv
│   │   ├── course_processed.csv
│   │   ├── ratings.csv
│   │   └── user_profile.csv
│   │
│   └── processed/
│       ├── courses_bows.csv
│       ├── sim.csv
│       ├── embedding_regression_results.csv
│       ├── embedding_classification_results.csv
│       └── collaborative_filtering_results.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_content_based_recommender.ipynb
│   ├── 03_clustering_recommender.ipynb
│   ├── 04_knn_collaborative_filtering.ipynb
│   ├── 05_nmf_collaborative_filtering.ipynb
│   ├── 06_neural_network_recommender.ipynb
│   ├── 07_embedding_models.ipynb
│   └── 08_model_evaluation.ipynb
│
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocess.py
│   │
│   ├── features/
│   │   ├── bow_features.py
│   │   ├── course_similarity.py
│   │   └── user_profiles.py
│   │
│   ├── models/
│   │   ├── clustering.py
│   │   ├── knn.py
│   │   ├── nmf.py
│   │   ├── neural_network.py
│   │   └── embedding_models.py
│   │
│   └── recommenders/
│       └── content_based.py
│
├── models/
│   ├── neural_network_recommender.keras
│   ├── user_embeddings.npy
│   └── item_embeddings.npy
│
├── reports/
│   ├── figures/
│   └── tables/
│
├── tests/
│
├── docs/
│
├── scripts/
│   └── setup_project.sh
│
├── requirements.txt
└── README.md
```

---

# Notebook Workflow

## 01 — Data Exploration

`01_data_exploration.ipynb`

This notebook investigates the structure and quality of the raw datasets.

Main activities:

- Load the four raw datasets.
- Inspect dataset dimensions.
- Examine data types.
- Check missing values.
- Check duplicate records.
- Analyze rating distributions.
- Analyze course interactions.
- Analyze user interests.
- Identify catalog coverage.

### Key finding

There are 307 courses in the catalog, but only 126 courses appear in the historical ratings data.

This creates a catalog-coverage limitation for collaborative filtering.

---

# 02 — Content-Based Recommender

`02_content_based_recommender.ipynb`

The first content-based approach uses user profiles and course genre features.

User profiles contain interest scores across different course categories. Course genre information provides corresponding course feature vectors.

Recommendation scores are calculated using the interaction between the user profile vector and course feature vector.

The general process is:

```text
User Profile
     │
     ▼
Genre Feature Vector
     │
     ▼
Compare with Course Feature Vectors
     │
     ▼
Recommendation Score
     │
     ▼
Top-N Courses
```

Courses already rated by the user are removed from the recommendation candidates.

---

# 03 — Clustering Recommender

`03_clustering_recommender.ipynb`

User profiles are standardized and analyzed using:

- StandardScaler
- PCA
- K-Means clustering

PCA is used to reduce dimensionality while retaining a specified proportion of variance.

K-Means clustering is then applied to identify groups of users with similar profile characteristics.

The clustering workflow is:

```text
User Profiles
     │
     ▼
Standardization
     │
     ▼
PCA
     │
     ▼
K-Means
     │
     ▼
User Clusters
```

The resulting clusters can be used to understand groups of users with similar interests.

---

# 04 — KNN Collaborative Filtering

`04_knn_collaborative_filtering.ipynb`

Collaborative filtering uses historical user-course interactions rather than course metadata.

The project initially considered user-based KNN. However, the dataset contains 33,901 users, making the user-user similarity matrix computationally expensive.

The implementation therefore uses **item-based KNN**.

Configuration:

- Algorithm: KNNBasic
- Similarity: Pearson
- Mode: Item-based
- Train/test split: 80/20
- Random state: 42

### Result

**RMSE: 0.1945**

Item-based KNN produced the lowest RMSE among the three collaborative-filtering recommender models tested.

---

# 05 — NMF Collaborative Filtering

`05_nmf_collaborative_filtering.ipynb`

Non-negative Matrix Factorization decomposes the user-item interaction matrix into lower-dimensional latent representations.

Configuration:

- Algorithm: NMF
- Number of factors: 15
- Train/test split: 80/20
- Random state: 42

### Result

**RMSE: 0.2075**

NMF performed slightly worse than item-based KNN on the observed test split.

---

# 06 — Neural Network Recommender

`06_neural_network_recommender.ipynb`

A neural-network recommender was implemented using separate user and item embedding layers.

The model uses:

- User embeddings
- Item embeddings
- 16-dimensional embeddings
- Dot-product interaction
- Adam optimizer
- Mean squared error loss

The architecture can be summarized as:

```text
User ID ──► User Embedding ──┐
                             ├──► Dot Product ──► Rating Prediction
Item ID ──► Item Embedding ──┘
```

Dataset dimensions used by the model:

- Users: 33,901
- Rated items: 126
- Interactions: 233,306
- Embedding dimension: 16

Train/test split:

- Training interactions: 186,644
- Test interactions: 46,662

The model was trained for 10 epochs with a batch size of 256.

### Result

**Test RMSE: 0.4076**

**Test MAE: 0.1617**

The neural-network result is specific to this project configuration and test split.

---

# 07 — Embedding Models

`07_embedding_models.ipynb`

The learned embeddings from the neural-network recommender were extracted and used as features for additional machine learning tasks.

Saved embedding dimensions:

```text
User embeddings:  (33901, 16)
Item embeddings:  (126, 16)
```

For each rating interaction, the corresponding user and item embeddings were combined using element-wise addition.

```text
User Embedding
       +
Item Embedding
       │
       ▼
16-dimensional Feature Vector
```

These features were used for both regression and classification.

---

## Embedding-Based Regression

The following regression models were evaluated:

- Linear Regression
- Ridge
- Lasso
- ElasticNet

### Results

| Model | RMSE | MAE |
|---|---:|---:|
| Linear Regression | 0.139921 | 0.062015 |
| Ridge | 0.139924 | 0.062006 |
| Lasso | 0.210945 | 0.089414 |
| ElasticNet | 0.210945 | 0.089414 |

Linear Regression produced the lowest RMSE, while Ridge produced an almost identical result and the lowest MAE.

---

## Embedding-Based Classification

The rating was converted into a binary classification target:

```text
Rating = 3  →  Class 1
Rating ≠ 3  →  Class 0
```

The resulting classes were highly imbalanced.

Two classification models were evaluated:

- Logistic Regression
- Random Forest

### Results

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.993635 | 0.994871 | 0.998471 | 0.996668 | 0.986744 |
| Random Forest | 0.994685 | 0.995364 | 0.999078 | 0.997218 | 0.991852 |

Random Forest produced the strongest result across all reported classification metrics on the test split.

---

# 08 — Model Evaluation

`08_model_evaluation.ipynb`

The final notebook consolidates and visualizes the model results.

## Collaborative Filtering Comparison

| Model | RMSE |
|---|---:|
| **Item-Based KNN** | **0.1945** |
| NMF | 0.2075 |
| Neural Network | 0.4076 |

Among the three collaborative-filtering recommender models, **Item-Based KNN achieved the lowest observed RMSE**.

## Embedding Regression Comparison

| Model | RMSE | MAE |
|---|---:|---:|
| **Linear Regression** | **0.139921** | 0.062015 |
| Ridge | 0.139924 | **0.062006** |
| Lasso | 0.210945 | 0.089414 |
| ElasticNet | 0.210945 | 0.089414 |

## Embedding Classification Comparison

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.993635 | 0.994871 | 0.998471 | 0.996668 | 0.986744 |
| **Random Forest** | **0.994685** | **0.995364** | **0.999078** | **0.997218** | **0.991852** |

---

# Technologies Used

### Programming

- Python
- Jupyter Notebook

### Data Processing

- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- Scikit-Surprise
- TensorFlow / Keras

### Natural Language Processing

- NLTK
- Gensim

### Visualization

- Matplotlib
- Seaborn

### Models and Techniques

- Bag-of-Words
- Cosine similarity
- PCA
- K-Means
- KNN collaborative filtering
- NMF
- Neural-network embeddings
- Linear Regression
- Ridge Regression
- Lasso
- ElasticNet
- Logistic Regression
- Random Forest

---

# Installation

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd ml-course-recommender
```

## 2. Create the Python environment

The project was developed using Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Download NLTK resources

Open Python and run:

```python
import nltk

nltk.download("stopwords")
```

## 5. Start Jupyter Notebook

```bash
jupyter notebook
```

Open the notebooks in numerical order.

---

# Reproducing the Project

Run the notebooks in the following order:

```text
01_data_exploration.ipynb
        ↓
02_content_based_recommender.ipynb
        ↓
03_clustering_recommender.ipynb
        ↓
04_knn_collaborative_filtering.ipynb
        ↓
05_nmf_collaborative_filtering.ipynb
        ↓
06_neural_network_recommender.ipynb
        ↓
07_embedding_models.ipynb
        ↓
08_model_evaluation.ipynb
```

The notebooks generate intermediate and processed artifacts in:

```text
data/processed/
```

Neural-network models and learned embeddings are stored in:

```text
models/
```

---

# Evaluation Metrics

The project uses different metrics depending on the task.

### Regression / rating prediction

- RMSE
- MAE

Lower values indicate lower prediction error.

### Classification

- Accuracy
- Precision
- Recall
- F1-score
- AUC

Because the classification target is highly imbalanced, F1-score, precision, recall, and AUC are considered alongside accuracy.

### Recommendation evaluation concepts

The project also studied recommendation-system metrics including:

- Precision
- Recall
- F1-score
- Diversity
- Novelty
- Coverage
- Click-through rate
- Serendipity
- Scalability

---

# Key Findings

1. **Item-based KNN performed best among the three collaborative-filtering recommender models**, achieving an RMSE of 0.1945.

2. **NMF achieved an RMSE of 0.2075**, slightly behind item-based KNN.

3. **The neural-network recommender achieved an RMSE of 0.4076** under the implemented configuration and test split.

4. **Embedding-based Linear Regression achieved an RMSE of 0.139921**, the lowest RMSE among the evaluated embedding regression models.

5. **Ridge regression performed almost identically to Linear Regression**, with RMSE of 0.139924 and MAE of 0.062006.

6. **Random Forest was the strongest embedding-based classifier** on the test split, achieving an F1-score of 0.997218 and AUC of 0.991852.

7. The dataset has a significant **catalog coverage limitation**, with only 126 of 307 catalog courses represented in historical ratings.

8. The classification target is highly imbalanced because most observed ratings are class 3.

---

# Limitations

The project has several limitations:

- Only a subset of the available course catalog has historical ratings.
- The rating values are limited to 2 and 3.
- The classification target is highly imbalanced.
- Collaborative filtering performance is affected by the limited number of rated items.
- The neural-network recommender was evaluated using the implemented architecture and training configuration; different architectures or hyperparameters could produce different results.
- The evaluation focuses primarily on rating prediction metrics rather than live user-interaction metrics such as CTR.

---

# Future Improvements

Potential future improvements include:

- Increasing catalog coverage.
- Incorporating additional user interaction data.
- Using richer course metadata.
- Applying hybrid recommendation methods.
- Performing systematic hyperparameter tuning.
- Addressing classification imbalance using appropriate techniques.
- Evaluating top-N recommendation quality using ranking metrics.
- Testing the recommender with real user feedback.
- Developing a production-ready recommendation API.
- Adding real-time recommendation capabilities.

---

# Final Project Deliverable

The final project is presented as:

**Machine Learning Capstone Project Report**

The presentation covers:

1. Introduction
2. Exploratory Data Analysis
3. Content-Based Recommender Using User Profile and Course Genres
4. Content-Based Recommender Using Course Similarity
5. Content-Based Recommender Using User Profile Clustering
6. KNN Collaborative Filtering
7. NMF Collaborative Filtering
8. Neural Network Embedding-Based Collaborative Filtering
9. Collaborative Filtering Algorithms Evaluation
10. Conclusion

Additional charts, tables, methodology details, and evaluation results can be included in the appendix.

---

# Author

**Tamarabrakemi Okpomu**

Machine Learning Capstone Project

---

## License

This project was developed for educational and academic purposes.