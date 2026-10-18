# Cross-Lingual Text Classification & Domain Prediction

A deep-learning-based cross-lingual domain classification web application built with **Flask**, **PyTorch**, and **Hugging Face Transformers** (`bert-base-multilingual-cased`).

The model classifies multilingual textual inputs into academic and general knowledge domains with confidence scoring.

---

## 🚀 Supported Classification Domains
The model predicts across 10 subject domains:
- **Biology**
- **Computer Science**
- **Countries**
- **Electrical Engineering**
- **General Topics**
- **Management Science**
- **Pakistan Studies**
- **Physics**
- **Psychology**
- **Zoology**

---

## 🛠️ Architecture & Tech Stack

- **Model Backbone:** `bert-base-multilingual-cased` / XLM-RoBERTa
- **Deep Learning Framework:** PyTorch & Transformers
- **Web Framework:** Flask
- **Preprocessing & Encoding:** Scikit-Learn (LabelEncoder) & HuggingFace Tokenizers
- **Deployment & Server:** Gunicorn / Flask Development Server

---

## 📂 Project Structure

```text
├── app.py                     # Flask web server and inference pipeline
├── requirements.txt           # Python dependencies
├── templates/
│   └── index.html             # Web UI for input and domain predictions
├── .gitignore                 # Git ignore rules
└── README.md                  # Project documentation
```

> **Note on Model Artifacts:**
> The trained weights (`crosslingual_artifact.pkl` / `crosslingual_model.pkl`) exceed GitHub's standard 100MB file limit and should be hosted via Git LFS, Hugging Face Model Hub, or external storage (e.g. Google Drive/S3).

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/WajahtAli/cross_lingual_text_classfication.git
cd cross_lingual_text_classfication
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download / Place Model Artifacts
Ensure `crosslingual_artifact.pkl` is placed in the project root directory.

### 5. Run the Application
```bash
python app.py
```

The application will start at `http://localhost:8000`.

---

## 💡 Usage

1. Open your browser and navigate to `http://localhost:8000`.
2. Enter multilingual or English text into the text area.
3. Click **Predict Domain** to receive the domain category along with the model's confidence percentage.
