# 🎬 Movie Review Classifier

A simple NLP project that classifies movie reviews as **Positive** or **Negative** using the **Naive Bayes** algorithm.

---

## 🧰 Libraries Used
| Library | Purpose |
|---|---|
| `scikit-learn` | Vectorization, model training & evaluation |
| `nltk` | NLP support |

---

## 🚀 How It Works
1. 40 labeled reviews (20 positive, 20 negative) are used as training data
2. `CountVectorizer` converts text into numerical data
3. `MultinomialNB` trains on 75% of the data and tests on 25%
4. Interactive mode lets you type your own review and get a prediction

---

## ⚙️ Setup & Run

```bash
pip install scikit-learn nltk
python movie_review_classifier.py
```

---

## 💬 Interactive Mode
## 💬 Interactive Mode
=== Movie Review Classifier ===

Enter your review: it was a good movie

Sentiment: Positive
Enter your review: boring and terrible acting

Sentiment: Negative
Enter your review: quit

Goodbye!
---

## 📊 Sample Output
Predictions: ['Positive', 'Positive', 'Negative', 'Negative', ...]

Actual:      ['Positive', 'Positive', 'Positive', 'Negative', ...]

Accuracy:    0.7
> Accuracy can be improved by expanding the dataset.

---

## 🛠️ Built With
Python 3.x · scikit-learn · nltk
