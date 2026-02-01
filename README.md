
# Classic NLP Sentiment Analysis

A **classic Natural Language Processing (NLP)** project implementing **sentiment analysis** using traditional machine learning techniques.  
This project intentionally avoids deep learning to demonstrate **core NLP fundamentals** used as baselines in industry and research.



---

## 📌 Problem Statement

Given an input text sentence, classify its sentiment into one of the following categories:

- **Positive**
- **Negative**

---

## 🧠 Approach

This project follows a standard pre–deep learning NLP pipeline :

- **Text Vectorization**: TF-IDF (Term Frequency–Inverse Document Frequency)
- **Model**: Multinomial Naive Bayes is chosen for its simplicity
- **Evaluation**: Stratified K-Fold Cross Validation

This approach is still relevant for :
- Baseline ML models
- Lightweight NLP systems
- Interpretable machine learning

---

## 🛠 Tech Stack

- **Language**: Python 3
- **Libraries**:
  - scikit-learn
  - pandas
  - numpy
  - joblib
- **Version Control**: Git, GitHub

---

## 📂 Project Structure

```
classic-nlp-sentiment/
├── data/
│   └── dataset.csv
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
├── model.joblib
├── vectorizer.joblib
├── requirements.txt
├── README.md
└── .gitignore
````

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Shadow-Coder-888/classic-nlp-sentiment.git
cd classic-nlp-sentiment
````

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Train the Model

```bash
python src/train.py
```

What happens :

* Stratified K-Fold cross-validation is performed
* Mean accuracy is printed
* Final model is trained on full dataset
* Model and vectorizer are saved to disk

---

## 🔮 Run Predictions

```bash
python src/predict.py
```

Example:

```
Enter text: i love this product
Prediction: positive
```

Type `exit` to stop.

---

## 📊 Evaluation Strategy

* Uses **Stratified K-Fold Cross Validation**
* Avoids unreliable single train–test split accuracy
* Suitable for small datasets

---

## ⚠️ Limitations

* No semantic understanding (classic NLP)
* Sensitive to spelling errors
* Performance depends heavily on dataset size
* Not suitable for complex language tasks

This project focuses on **correct methodology**, not state-of-the-art accuracy.

---

## 🔮 Future Improvements

* Larger real-world dataset
* Spell normalization
* Stemming / lemmatization
* Precision, recall, F1-score
* REST API using FastAPI
* Deep learning upgrade (Transformers)

---

