# 📰 News Topic Classifier Using BERT

A fine-tuned BERT model that classifies news headlines and articles into one of four topic categories: **World**, **Sports**, **Business**, and **Sci/Tech**. Built with Hugging Face Transformers and deployed as an interactive Streamlit app.

## 🔗 Live Demo

Try the deployed app here: **[zknewsclassifier.streamlit.app](https://zknewsclassifier.streamlit.app/)**

## 🎯 Problem Statement

News platforms handle large volumes of articles daily and need an automated way to categorize them by topic. Manually tagging each article is slow and doesn't scale, creating a need for a model that can classify news text accurately and in real time.

## 🚀 Goal

Fine-tune a pretrained transformer model (`bert-base-uncased`) on the AG News dataset to classify news text into four categories, then deploy it as an interactive web app for live predictions.

## 📊 Results

| Metric | Score |
|---|---|
| Accuracy | 94.6% |
| F1-score (macro) | 94.6% |

The model was validated through CLI testing on both dataset samples and custom headlines, and performs best on longer, news-style phrasing (consistent with its training data format).

## 🧠 Approach

1. **Dataset**: [AG News](https://huggingface.co/datasets/sh0416/ag_news) — 120,000 training and 7,600 test examples across 4 balanced classes.
2. **Preprocessing**: Merged `title` and `description` fields into a single input text, tokenized with BERT's tokenizer (max length 128), and aligned labels to a 0-indexed range.
3. **Fine-tuning**: Trained `bert-base-uncased` for 3 epochs using Hugging Face's `Trainer` API with a learning rate of 2e-5.
4. **Evaluation**: Measured accuracy and macro F1-score on the held-out test set.
5. **Deployment**: Built a Streamlit app for live text classification, with the fine-tuned model hosted on the Hugging Face Hub.

## 🛠️ Tech Stack

- Python
- Hugging Face Transformers & Datasets
- PyTorch
- Streamlit
- Hugging Face Hub

## 📁 Project Structure

```
├── app.py                    # Streamlit deployment app
├── train_bert_agnews.py      # Model fine-tuning script
├── evaluate_model.py         # Evaluation script (accuracy, F1, confusion matrix)
├── requirements.txt          # Project dependencies
└── README.md
```

## 💻 Running Locally

```bash
git clone https://github.com/ZainAliKhanZK/bert-agnews-classifier.git
cd bert-agnews-classifier
pip install -r requirements.txt
streamlit run app.py
```

The app loads the fine-tuned model directly from the Hugging Face Hub, so no local model files are needed.

## 🤗 Model

The fine-tuned model is available on the Hugging Face Hub: [ZainAliKhanZK/bert_agnews_classifier](https://huggingface.co/ZainAliKhanZK/bert_agnews_classifier)

## 📌 Categories

| Emoji | Category |
|---|---|
| 🌍 | World |
| 🏅 | Sports |
| 💼 | Business |
| 🔬 | Sci/Tech |

## 👤 Author

**Zain Ali Khan**
[GitHub](https://github.com/ZainAliKhanZK) · [Hugging Face](https://huggingface.co/ZainAliKhanZK)
