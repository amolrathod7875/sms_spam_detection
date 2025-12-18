
# SMS Spam Detection with Machine Learning

A high-performance machine learning model that classifies SMS messages as either "Spam" or "Ham" (legitimate) using Natural Language Processing (NLP) techniques and the Multinomial Naive Bayes algorithm.

## 📌 Project Overview

The goal of this project is to build a prediction model that accurately classifies text messages to filter out unwanted spam. Using the **SMS Spam Collection Dataset**, the model was trained to distinguish between spam and legitimate messages with high precision, ensuring that important messages are not mislabeled.

### 🏆 Key Results

The model achieved exceptional performance metrics, making it highly reliable for spam filtering tasks:

* **Accuracy:** 98.92%
* **Precision:** 94.35% *(Crucial for minimizing false positives)*
* **Recall:** 93.84%
* **F1 Score:** 94.08%

## 📂 About the Dataset

**Source:** [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)

The dataset consists of **5,574 SMS messages** in English, tagged as `ham` (legitimate) or `spam`.

* **v1**: Label (ham/spam)
* **v2**: Raw text message

### Dataset Composition & Sources

The corpus was compiled from various free-for-research sources:

1. **Grumbletext Website:** 425 spam messages manually extracted from a UK forum where users report spam.
2. **NUS SMS Corpus (NSC):** 3,375 randomly chosen legitimate (ham) messages from the Department of Computer Science at the National University of Singapore.
3. **Caroline Tag's PhD Thesis:** 450 ham messages.
4. **SMS Spam Corpus v.0.1 Big:** 1,002 ham messages and 322 spam messages.

*Acknowledgements:*

> The original dataset can be found [here](http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/).
> **Citation:** Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. Contributions to the Study of SMS Spam Filtering: New Collection and Results. Proceedings of the 2011 ACM Symposium on Document Engineering (DOCENG'11), Mountain View, CA, USA, 2011.

## 🛠️ Technologies Used

* **Python**: Core programming language.
* **Pandas & NumPy**: Data manipulation and analysis.
* **Scikit-Learn**: Machine learning (`MultinomialNB`) and model evaluation.
* **NLTK (Natural Language Toolkit)**: Text preprocessing (tokenization, stopwords).
* **Matplotlib & Seaborn**: Data visualization.

## 🚀 How to Run

1. **Clone the repository:**
```bash
git clone https://github.com/amolrathod7875/sms_spam_detection.git

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Run the Notebook:**
```bash
jupyter notebook "SMS Spam Detection with Machine Learning.ipynb"

```



## 📊 Methodology

1. **Data Cleaning:** Handling null values and mapping labels (`ham`=0, `spam`=1).
2. **Text Preprocessing:** Converting to lowercase, removing punctuation, and tokenization.
3. **Feature Extraction:** Using `CountVectorizer` (Bag of Words) to convert text into numerical vectors.
4. **Model Training:** Implementing the **Multinomial Naive Bayes** classifier, which is highly effective for text data.
5. **Evaluation:** Measuring success via Accuracy, Precision, Recall, and F1-Score.

---

*Created by Amol Rathod*