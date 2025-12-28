# Diabetes Prediction System

A machine learning-powered web application that predicts the likelihood of diabetes based on medical diagnostics. The system uses a Random Forest Classifier trained on the PIMA Indians Diabetes Dataset and provides a user-friendly, responsive web interface.

## Features

- **Accurate Predictions**: Uses a trained Machine Learning model (Random Forest) for reliable predictions.
- **Interactive Web Interface**: Built with Flask, featuring a clean, responsive design.
- **Real-time Analysis**: Instant feedback with a probability/confidence score.
- **Data Preprocessing**: Handles missing values and scales data for better model performance.
- **Responsive Design**: Works seamlessly on desktop, tablets, and mobile devices.

## Technologies Used

- **Backend**: Python, Flask
- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Frontend**: HTML5, CSS3 (Responsive, Custom Theme)

## Installation

1.  **Clone the repository** (or download the files):
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2.  **Create a Virtual Environment** (Recommended):
    ```bash
    python -m venv .venv
    # Windows
    .\.venv\Scripts\activate
    # Mac/Linux
    source .venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Train the Model** (Optional - if `model.pkl` is missing):
    ```bash
    python train_model.py
    ```
    This will generate `model.pkl` and `scaler.pkl`.

2.  **Run the Application**:
    ```bash
    python app.py
    ```

3.  **Access the Website**:
    Open your browser and navigate to: `http://127.0.0.1:5000`

4.  **Get a Prediction**:
    Enter the required medical details (Pregnancies, Glucose, Blood Pressure, etc.) and click "Analyze Report".

## Dataset

The model is trained on the [PIMA Indians Diabetes Dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database).
