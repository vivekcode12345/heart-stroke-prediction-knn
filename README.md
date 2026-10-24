# ❤️ Heart Stroke Prediction — KNN Classifier

A machine learning web application that predicts the risk of heart disease using a **K-Nearest Neighbors (KNN)** classifier. Built with **Python**, **Streamlit**, **scikit-learn**, and **pandas**.

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 Overview

Heart disease is one of the leading causes of mortality worldwide. Early detection can significantly improve outcomes. This project provides an interactive tool where users input their health metrics and receive an instant prediction of whether they are at **high risk** or **low risk** of heart disease, powered by a trained KNN model.

### What it does
- Accepts user health inputs via a friendly Streamlit UI.
- Preprocesses and scales the input using a saved scaler.
- Runs the input through a trained KNN classifier.
- Displays a clear risk prediction result.

---

## 🗂️ Project Structure

```
Heart/
├── app.py                # Streamlit web application
├── Heart.ipynb           # Jupyter notebook (EDA, training, model export)
├── heart.csv             # Dataset used for training
├── KNN_heart.pkl         # Trained KNN model (pickle)
├── scaler.pkl            # Fitted feature scaler (pickle)
├── columns.pkl           # Expected model input columns (pickle)
├── LICENSE               # MIT License
├── README.md             # Project documentation
└── .gitignore            # Git ignore rules
```

---

## 📊 Dataset

The model is trained on `heart.csv`, which contains the following features:

| Feature | Description |
|---------|-------------|
| `Age` | Age of the patient |
| `Sex` | Gender (M / F) |
| `ChestPainType` | Type of chest pain (ATA, NAP, TA, ASY) |
| `RestingBP` | Resting blood pressure (mm Hg) |
| `Cholesterol` | Serum cholesterol (mg/dL) |
| `FastingBS` | Fasting blood sugar > 120 mg/dL (1 = yes, 0 = no) |
| `RestingECG` | Resting electrocardiogram results (Normal, ST, LVH) |
| `MaxHR` | Maximum heart rate achieved |
| `ExerciseAngina` | Exercise-induced angina (Y / N) |
| `Oldpeak` | ST depression induced by exercise |
| `ST_Slope` | Slope of the peak exercise ST segment (Up, Flat, Down) |
| `HeartDisease` | Target — presence of heart disease (1 = yes, 0 = no) |

---

## 🛠️ Tech Stack

- **Python 3**
- **Streamlit** — Web UI
- **pandas** — Data handling
- **scikit-learn** — KNN classifier, preprocessing, scaling
- **joblib** — Model/scaler serialization
- **Jupyter Notebook** — Exploration & training

---

## 🚀 Getting Started

### Prerequisites

Python 3.8+ is recommended.

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/vivekcode12345/heart-stroke-prediction-knn.git
   cd heart-stroke-prediction-knn
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS/Linux
   # venv\Scripts\activate         # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install streamlit pandas scikit-learn joblib
   ```

   *(If a `requirements.txt` is added later: `pip install -r requirements.txt`)*

### Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🧪 Model Details

- **Algorithm:** K-Nearest Neighbors (KNN) Classifier
- **Serialization:** Model, scaler, and expected columns are saved via `joblib` as `.pkl` files
- **Inference flow:**
  1. User enters health metrics in the Streamlit UI
  2. Input is converted to a one-hot encoded DataFrame matching training columns
  3. Missing columns are filled with `0`
  4. Data is scaled with the pre-fitted scaler
  5. The KNN model predicts `0` (Low Risk) or `1` (High Risk)

---

## 📸 Usage

1. Launch the app with `streamlit run app.py`.
2. Adjust the sliders and select options for your health parameters.
3. Click **Predict**.
4. View the result:
   - 🔴 **High Risk of Heart Disease**
   - 🟢 **Low Risk of Heart Disease**

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m "Add your feature"`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Vivek Verma**  
GitHub: [@vivekcode12345](https://github.com/vivekcode12345)

---

> ⚠️ **Disclaimer:** This tool is for educational and informational purposes only. It is **not a substitute** for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider regarding any medical concerns.