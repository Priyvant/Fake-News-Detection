# Fake News Detection

A machine learning project that classifies news articles as real or fake
using natural language processing and classification models.

## Problem Statement

Misinformation spreads fast on digital platforms and can have serious
real-world consequences. This project builds and compares multiple
classifiers to automatically flag news articles that are likely to be
fake, based on the text content of the article.

## Dataset

[Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
(Kaggle) — ~44,900 labeled news articles, each marked as real (1) or
fake (0). Only the article `text` column was used for modeling; `title`,
`subject`, and `date` were dropped.

## Approach

1. **Text preprocessing** — removed punctuation, lowercased text, and
   stripped stopwords using NLTK
2. **Exploratory analysis** — generated word clouds for real vs. fake
   articles, and a bar chart of the most frequent words, to see how the
   two classes differ in language
3. **Feature extraction** — converted the cleaned text into numerical
   features using **TF-IDF** (`TfidfVectorizer`)
4. **Model training** — trained and compared four classifiers:
   - Logistic Regression
   - Decision Tree Classifier
   - Gradient Boosting Classifier
   - Random Forest Classifier
5. **Evaluation** — compared train vs. test accuracy for each model
   (to check for overfitting) and used confusion matrices to see the
   error breakdown

## Results

| Model                       | Train Accuracy | Test Accuracy |
|------------------------------|-----------------|----------------|
| Logistic Regression          | 99.40%          | 98.75%         |
| Decision Tree Classifier     | 100%            | 99.62%         |
| Gradient Boosting Classifier | 99.70%          | 99.55%         |
| Random Forest Classifier     | 99.99%          | 98.94%         |

**Best model: Decision Tree Classifier**, with 99.62% test accuracy,
closely followed by Gradient Boosting at 99.55%.

Both Decision Tree and Random Forest hit near-100% train accuracy,
which is a classic overfitting signal — they're memorizing the training
data almost perfectly. Gradient Boosting shows the smallest gap between
train and test accuracy (0.15 points), making it the most "honest"
model of the four in terms of generalization, even though Decision Tree
edges it out slightly on raw test accuracy.

The Decision Tree confusion matrix on the test set shows 5,738 correct
"Fake" predictions against only 18 misclassifications, and 5,451 correct
"Real" predictions against 23 misclassifications — very few errors
either way.

## Project Structure

```
Fake-News-Detection/
├── Fake News Detection using Machine Learning.ipynb   # main notebook
├── requirements.txt
└── README.md
```

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Open the notebook
jupyter notebook "Fake News Detection using Machine Learning.ipynb"

# 3. Run all cells (make sure the dataset CSV is in the same folder,
#    or update the file path in the notebook)
```

## Tech Stack

Python, Pandas, Scikit-learn, NLTK, WordCloud, Seaborn, Matplotlib

## What I Learned

- How text needs to be cleaned (stopword removal, punctuation stripping)
  and vectorized (TF-IDF) before it can be fed into a classical ML model
- Word clouds and word-frequency charts are a quick way to sanity-check
  whether the two classes actually differ in vocabulary before even
  training a model
- Comparing train vs. test accuracy side by side is a simple but
  effective way to catch overfitting — Decision Tree and Random Forest
  both hit near-100% train accuracy, a signal that they're closer to
  memorizing the data than generalizing from it
- Higher test accuracy alone doesn't tell the whole story — Gradient
  Boosting had a smaller train-test gap than Decision Tree despite a
  marginally lower test score, which matters more if this model were
  ever used on genuinely new, unseen articles
- Comparing multiple classifiers systematically instead of committing to
  one algorithm upfront

## Future Improvements

- Try a more powerful text representation (word embeddings, or a
  transformer-based model like BERT) and compare against the classical
  models here
- Deploy the best model as a simple web app (Streamlit/Flask) where a
  user can paste an article and get a prediction
- Test the model on more recent news articles to check how well it
  generalizes beyond the training dataset's time period

