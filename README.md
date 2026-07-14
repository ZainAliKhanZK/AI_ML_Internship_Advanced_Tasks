# 🤖 AI/ML Engineering – Advanced Internship Projects

A portfolio of advanced machine learning and AI projects completed as part of the **AI/ML Engineering Internship at DevelopersHub Corporation**, covering transformer fine-tuning, production ML pipelines, multimodal learning, and conversational AI with LLMs and retrieval-augmented generation.

Each project below is self-contained, with its own repository, README, and (where applicable) a live demo.

---

## 📰 1. News Topic Classifier Using BERT

Fine-tuned `bert-base-uncased` on the AG News dataset to classify news headlines into **World, Sports, Business, and Sci/Tech** categories, deployed as a live Streamlit app.

- **Result:** 94.6% accuracy / 94.6% macro F1-score on the held-out test set
- **Tech:** Hugging Face Transformers & Datasets, PyTorch, Streamlit, Hugging Face Hub
- 🔗 [Live Demo](https://zknewsclassifier.streamlit.app/) · 🤗 [Model](https://huggingface.co/ZainAliKhanZAK/bert_agnews_classifier) · 📂 [Repository](https://github.com/ZainAliKhanZK/bert-agnews-classifier)

---

## 📊 2. Customer Churn Prediction Pipeline

An end-to-end, production-ready ML pipeline using Scikit-learn's `Pipeline` API to predict telecom customer churn, with automated preprocessing, model comparison, hyperparameter tuning, and export via `joblib`.

- **Result:** Logistic Regression selected — 80.6% accuracy, 0.60 F1-score on the churn class (outperforming Random Forest)
- **Tech:** Scikit-learn (Pipeline, ColumnTransformer, GridSearchCV), pandas, joblib
- 📂 [Repository](https://github.com/ZainAliKhanZK/churn-prediction-pipeline)

---

## 🏠 3. Multimodal Housing Price Prediction (Images + Tabular Data)

A multimodal regression model that fuses CNN-extracted image features (via a frozen, pretrained MobileNetV2) with structured tabular data to predict housing prices — combining visual and numerical signals in a single pipeline.

- **Result:** MAE of $223,040.09 / RMSE of $308,576.54, with strong accuracy on individual cases (e.g. within ~1% on a $699,999 listing)
- **Tech:** TensorFlow/Keras (MobileNetV2, transfer learning), OpenCV, Scikit-learn
- 📂 [Repository](https://github.com/ZainAliKhanZK/multimodal-housing-price-prediction)

---

## 📚 4. Context-Aware RAG Chatbot

A conversational chatbot built with LangChain and ChromaDB that retrieves answers from a custom Wikipedia-based knowledge base while maintaining multi-turn conversational memory, deployed with live switching between three LLM providers.

- **Result:** Verified accurate retrieval and correct pronoun/context resolution across turns; deployed with switchable Groq, Gemini, and Hugging Face backends, each with independent memory and source-attributed answers
- **Tech:** LangChain, ChromaDB, `sentence-transformers/all-MiniLM-L6-v2`, Streamlit, Groq/Gemini/Hugging Face APIs
- 📂 [Repository](https://github.com/ZainAliKhanZK/rag-chatbot)

---

## 🛠️ Core Skills Demonstrated

| Area | Skills |
|---|---|
| NLP & Transformers | Transfer learning, fine-tuning, tokenization, evaluation metrics |
| ML Engineering | Pipeline design, hyperparameter tuning, model export & reusability |
| Computer Vision | CNN feature extraction, multimodal feature fusion |
| Conversational AI | RAG, vector embeddings, LLM integration, conversational memory |
| Deployment | Streamlit, Hugging Face Hub, live interactive demos |

## 👤 Author

**Zain Ali Khan**
[GitHub](https://github.com/ZainAliKhanZK) · [Hugging Face](https://huggingface.co/ZainAliKhanZK)
