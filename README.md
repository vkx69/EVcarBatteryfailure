# 🔋 EV Battery Failure Prediction

A machine-learning powered Streamlit application for predicting **electric vehicle battery failure risk** using vehicle, battery, charging, driving, and environmental parameters.

## 🚀 Features

* 🔋 EV battery failure prediction
* 🤖 Random Forest machine-learning model
* 📊 Failure probability and prediction results
* ⚡ Battery and charging analytics
* 🚗 Vehicle telemetry inputs
* 📈 Interactive Streamlit dashboard
* 💾 Trained model stored with Git LFS
* 📐 StandardScaler preprocessing

## 🧠 Machine Learning Model

The application uses a trained **Random Forest Classifier** with:

* **156 input features**
* Binary classification
* `battery_model.pkl` — trained model
* `scaler.pkl` — feature scaler

The model receives the same feature structure used during training to ensure consistent predictions.

## 📁 Project Structure

```text
EVcarBatteryfailure/
│
├── app.py
├── battery_model.pkl
├── scaler.pkl
├── requirements.txt
├── EVCar.ipynb
├── ev_battery_failure_dataset.csv
├── .gitattributes
└── README.md
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/vkx69/EVcarBatteryfailure.git
cd EVcarBatteryfailure
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

## 📦 Dependencies

Main Python packages:

```text
streamlit
pandas
numpy
scikit-learn==1.6.1
joblib
```

## 💾 Git LFS

The trained `battery_model.pkl` file is larger than GitHub's normal 100 MB file limit, so **Git Large File Storage (Git LFS)** is used.

Install Git LFS:

```bash
git lfs install
```

Check tracked LFS files:

```bash
git lfs ls-files
```

## ☁️ Streamlit Deployment

The application can be deployed using Streamlit Cloud.

Select:

```text
Repository: vkx69/EVcarBatteryfailure
Branch: main
Main file: app.py
```

Streamlit will install dependencies from `requirements.txt` and load the model files from the repository.

## ⚠️ Disclaimer

This application provides a **machine-learning based prediction** and should be considered a decision-support tool. It is not a certified battery safety system, engineering inspection, or guarantee of battery failure.

## 👨‍💻 Author

**Vikas Kumar**

GitHub: https://github.com/vkx69
