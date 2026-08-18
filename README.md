# Flood Resilience & Weather Prediction System 🌊🛰️

An AI-powered disaster management system designed to predict localized flood risks and automate emergency response dispatches. This project utilizes machine learning to analyze real-time environmental metrics and provides an interactive dashboard for risk monitoring and resilience planning.

---

## 🚀 Features

* **Predictive Analytics:** Uses an optimized **XGBoost Classifier** to calculate precise flood probabilities based on Rainfall, Humidity, and Temperature metrics.
* **Real-Time Automated Alerts:** Integrated with **Twilio API** to dispatch instant SMS emergency warnings to local authorities and residents when risk thresholds are breached.
* **Interactive Dashboard:** Built with **Streamlit** for real-time parameter tweaking, live risk level visualization, and interactive decision support.
* **Data-Driven Insights:** Capable of handling large-scale historical weather datasets for robust model training and environmental trend analysis.
* **Credential Isolation:** Designed using security best practices to keep sensitive API keys isolated from public version control.

---

## 🧠 Model Details

This system utilizes machine learning trained on **500,000+ environmental records** with the following specification:

* **Algorithm:** XGBoost (Extreme Gradient Boosting) Classifier
* **Primary Inputs:** Rainfall, Temperature, Humidity
* **Output:** Risk Probability % & Categorical Threat Level (Low / Moderate / Extreme)
* **Alert Logic:** Threshold engine triggering automated SMS broadcast upon extreme threat detection

---

## 🏗 Architecture

```text
flood-resilience-system/
├── app.py              ← Streamlit dashboard & Twilio dispatch logic
├── train.py            ← ML model training script
├── preprocessing.py    ← Data cleaning & feature engineering
├── flood_model.json    ← Exported XGBoost model artifact
├── train.csv           ← Historical training dataset (500,000+ records)
├── config.py           ← Local API credentials (ignored by Git)
├── .gitignore          ← Security rules for sensitive files
├── requirements.txt    ← Project dependencies
```
---

## 🛠 Tech Stack

* **Language:** Python 3.10+
* **Machine Learning:** XGBoost, Scikit-learn
* **Data Processing:** Pandas, NumPy
* **Web Framework:** Streamlit
* **Messaging API:** Twilio REST API

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone [https://github.com/kishoredammala0320/flood-resilience-system.git](https://github.com/kishoredammala0320/flood-resilience-system.git)
cd flood-resilience-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
### 2. Configure Local Credentials

Option A: Local Environment (config.py)
Create a config.py file in the root directory (make sure it stays in .gitignore):

```python
OPENWEATHER_API_KEY = "your_actual_weather_key"
TWILIO_SID = "your_actual_twilio_sid"
TWILIO_AUTH_TOKEN = "your_actual_twilio_token"
TWILIO_NUMBER = "your_actual_twilio_number"
TARGET_PHONE = "your_actual_recipient_number"
```
Option B: Streamlit Community Cloud (Deployment)
If deploying to Streamlit Cloud, do not upload config.py. Instead, go to App Settings -> Secrets and paste:

```python
OPENWEATHER_API_KEY = "your_actual_weather_key"
TWILIO_SID = "your_actual_twilio_sid"
TWILIO_AUTH_TOKEN = "your_actual_twilio_token"
TWILIO_NUMBER = "your_actual_twilio_number"
TARGET_PHONE = "your_actual_recipient_number"
```
### 3. Run the Dashboard
```Bash
streamlit run app.py
```
### 4. Open in Browser
```Plaintext
http://localhost:8501
```
Note: Ensure config.py is listed in your .gitignore file before making any Git commits to keep your API keys secure.
