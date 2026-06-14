# Plant Health Monitor

A production-oriented Streamlit dashboard that screens plant leaf images as
**Healthy** or **Diseased** using traditional computer vision and machine
learning. It intentionally does not use TensorFlow, Keras, or neural networks.

## Features

- Modern, responsive agriculture-themed interface
- JPG, JPEG, and PNG image upload with preview
- OpenCV preprocessing in RGB and HSV color spaces
- Color, histogram, texture, sharpness, and edge-density feature extraction
- Random Forest, SVM, and Logistic Regression model selection
- Automatic model training and persistence on first startup
- Prediction confidence, health score, status, and recommendations
- Model metrics, confusion matrix, feature influence, and history charts
- Custom feature CSV validation and model retraining
- Downloadable prediction-history CSV and per-image PDF report

## Project Structure

```text
plant-health-monitor/
|-- app.py
|-- dataset/
|-- images/
|-- models/
|-- reports/
|-- utils/
|   |-- __init__.py
|   |-- preprocessing.py
|   |-- features.py
|   `-- prediction.py
|-- requirements.txt
`-- README.md
```

## Quick Start

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the virtual environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies and start the dashboard:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## How It Works

1. The uploaded image is decoded by OpenCV and resized to `256 x 256`.
2. RGB and HSV representations are generated.
3. The app extracts 35 numeric features:
   - Mean RGB and HSV values
   - Eight-bin histograms for red, green, and blue
   - Grayscale mean and standard deviation
   - Grayscale entropy
   - Laplacian variance
   - Canny edge density
4. The selected scikit-learn classifier predicts Healthy (`0`) or Diseased (`1`).
5. The dashboard displays confidence, a health score, and a general recommendation.

## Dataset Format

The Dataset page accepts CSV files with every feature column listed by
`utils/features.py` plus a `label` column:

- `0` or `Healthy` for healthy leaves
- `1` or `Diseased` for diseased leaves

Use **Download dataset template** in the app to obtain an exact schema.
At least 20 valid rows and examples from both classes are required.

## Initial Model Training

When a model file does not exist, the application trains it automatically on a
reproducible synthetic feature dataset. This fallback makes the project runnable
without bundling a large image dataset. It is for demonstration only.

For meaningful results, collect representative labeled leaf images, extract the
same features, upload the resulting CSV, and retrain each model from the Dataset
page. Saved model bundles are written to `models/`.

## Deployment

The project works with Streamlit Community Cloud and container-based platforms.
Use this start command:

```bash
streamlit run app.py --server.address 0.0.0.0
```

The app creates `dataset/`, `models/`, and `reports/` if they do not exist.
For deployments with an ephemeral filesystem, saved models will be retrained
after a restart unless persistent storage is configured.

## Important Limitation

This application provides a visual screening result, not a confirmed botanical
diagnosis. Image conditions, plant species, disease stage, and dataset quality
all affect accuracy. Confirm consequential treatment decisions with a qualified
agriculture or horticulture professional.
