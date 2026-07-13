# 🏠 Multimodal Housing Price Prediction (Images + Tabular Data)

A multimodal machine learning model that predicts housing prices by combining CNN-extracted image features with structured tabular data — bridging visual and numerical signals into a single prediction pipeline.

## 🎯 Problem Statement

Housing prices depend on both measurable attributes (bedrooms, bathrooms, square footage) and visual factors (interior condition, layout, aesthetic appeal) that structured data alone can't capture. Relying on tabular features only misses valuable visual signals that influence real-world pricing.

## 🚀 Goal

Build a multimodal model that combines CNN-extracted image features with structured tabular data to predict housing prices, using a pretrained CNN for image feature extraction and a regression model trained on the fused feature set, evaluated using MAE and RMSE.

## 📊 Results

| Metric | Score |
|---|---|
| MAE | $223,040.09 |
| RMSE | $308,576.54 |

**Example prediction:** for a house with an actual price of $699,999, the model predicted $706,841.95 — a difference of just $6,842.95, showing strong accuracy on individual cases even with a wider average error across the full test set.

## 🧠 Approach

1. **Dataset**: [Houses Dataset](https://github.com/emanhamed/Houses-dataset) (Ahmed & Moustafa) — 535 house listings, each with 4 images (bedroom, bathroom, kitchen, frontal view) and tabular attributes (bedrooms, bathrooms, area, zipcode, price).
2. **Image preprocessing**: Combined each house's 4 images into a single 2×2 montage image, resized for CNN input.
3. **Feature extraction**: Used a pretrained **MobileNetV2** (frozen, ImageNet weights, no top layer) to extract a fixed-length feature vector from each house's image montage — no training from scratch, avoiding overfitting on a small image dataset.
4. **Tabular preprocessing**: Scaled continuous features (bedrooms, bathrooms, area) with `MinMaxScaler` and one-hot encoded categorical features (zipcode).
5. **Multimodal fusion**: Concatenated CNN-extracted image features with the processed tabular features into a single combined feature vector per house.
6. **Modeling**: Trained a regression model on the fused feature set to predict house price.
7. **Evaluation**: Assessed performance using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras (MobileNetV2, transfer learning)
- OpenCV (image loading & preprocessing)
- Scikit-learn (preprocessing, regression model, evaluation metrics)
- pandas, NumPy

## 📁 Project Structure

```
├── housing_price_pipeline.py   # Full pipeline: preprocessing, feature extraction, training, evaluation
├── requirements.txt
└── README.md
```

## 💻 Usage

**Setup:**
```bash
git clone https://github.com/ZainAliKhanZK/multimodal-housing-price-prediction.git
cd multimodal-housing-price-prediction
pip install -r requirements.txt
```

**Run the pipeline** (recommended on Google Colab with GPU, due to CNN feature extraction):
```bash
python housing_price_pipeline.py
```

## ⚠️ Notes & Limitations

- Trained on a relatively small dataset (535 houses), which limits generalization, especially for high-value or unusual properties.
- The gap between MAE and RMSE suggests a subset of predictions carry larger errors, likely concentrated around higher-priced homes — a common pattern in price prediction with limited training data.
- New input data (images and tabular features) must follow the exact same preprocessing steps and column structure as training data for accurate predictions.
- Potential future improvements: fine-tuning the CNN backbone instead of using it purely as a frozen feature extractor, using a larger/more diverse image dataset, or exploring a joint neural network architecture instead of feature concatenation + a separate regressor.

## 👤 Author

**Zain Ali Khan**
[GitHub](https://github.com/ZainAliKhanZK) · [Hugging Face](https://huggingface.co/ZainAliKhanZK)
