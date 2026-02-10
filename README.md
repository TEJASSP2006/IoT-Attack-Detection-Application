# IoT Attack Detection Application

A comprehensive machine learning-based application for detecting and classifying IoT network attacks using the UNB CIC IoT Dataset 2023.

## 🎯 Overview

This application can detect **7 types of IoT attacks**:

1. **DDoS** - Distributed Denial of Service
2. **DoS** - Denial of Service
3. **Recon** - Reconnaissance attacks
4. **Web-based** - Web application attacks
5. **Brute Force** - Password/authentication attacks
6. **Spoofing** - Identity spoofing attacks
7. **Mirai** - Mirai botnet attacks

Plus detection of **Benign** (normal) traffic.

## 📊 Dataset Information

The application uses the **UNB CIC IoT Dataset 2023**, which includes:
- **33 different attacks** across 7 categories
- Data collected from **105 IoT devices**
- Real IoT devices as both attackers and victims
- **47 network traffic features**

### Dataset Download

1. Visit: https://www.unb.ca/cic/datasets/iotdataset-2023.html
2. Download the CSV dataset files
3. You may also find the dataset on Kaggle: https://www.kaggle.com/datasets/madhavmalhotra/unb-cic-iot-dataset

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM (recommended for training)

### Installation

1. **Clone or download this project**

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install flask pandas numpy scikit-learn joblib matplotlib seaborn
```

### Step 1: Train the Model

1. Download the CIC IoT 2023 dataset (CSV format)

2. Edit `iot_attack_detector.py` and update the dataset path:
```python
dataset_path = "path/to/your/dataset.csv"
```

3. Uncomment the training code in the `main()` function (lines are marked)

4. Run the training script:
```bash
python iot_attack_detector.py
```

This will:
- Load and preprocess the dataset
- Train a Random Forest classifier
- Evaluate the model performance
- Display feature importance
- Save the trained model as `iot_attack_detector.pkl`

**Expected Training Time**: 5-15 minutes depending on dataset size and hardware

### Step 2: Launch the Web Dashboard

Once the model is trained:

```bash
python web_dashboard.py
```

Access the dashboard at: **http://localhost:5000**

## 🖥️ Using the Application

### Web Dashboard Features

The dashboard has **3 main tabs**:

#### 1. Single Prediction
- Analyze individual network traffic samples
- Input features as JSON array
- Get real-time attack classification
- View probability distribution across all attack types

**Example Usage**:
- Click "Generate Random Sample" for testing
- Or paste actual network traffic features
- Click "Analyze Traffic"

#### 2. Batch Analysis
- Upload CSV files with multiple traffic samples
- Analyze hundreds or thousands of samples at once
- Get comprehensive statistics and summary
- View first 10 predictions with confidence scores

**CSV Format**:
- Must contain all required features (same as training data)
- No header row needed if features match training order

#### 3. Live Monitor
- Real-time statistics dashboard
- Attack type distribution charts
- Recent detection log (last 10)
- Auto-refresh every 5 seconds

## 📁 Project Structure

```
iot-attack-detection/
│
├── iot_attack_detector.py      # Core ML model and training
├── web_dashboard.py             # Flask web server
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── templates/
│   └── dashboard.html          # Web interface
│
└── iot_attack_detector.pkl     # Trained model (generated after training)
```

## 🔧 Model Architecture

### Random Forest Classifier
- **Estimators**: 100 trees
- **Max Depth**: 20
- **Features**: 46-47 network traffic features
- **Classes**: 8 (7 attacks + Benign)

### Key Features Used
The model uses network traffic features including:
- Flow duration
- Packet statistics (count, length, rate)
- Protocol information
- Flags and headers
- Inter-arrival times
- Port numbers
- And more...

## 📈 Performance

Typical model performance (varies by dataset):
- **Accuracy**: 95-99%
- **Precision**: 94-98% (per attack class)
- **Recall**: 93-97% (per attack class)
- **F1-Score**: 94-98% (per attack class)

Performance metrics are displayed after training.

## 🔍 API Endpoints

The web application provides REST API endpoints:

### POST /api/predict
Predict attack type for single sample
```json
{
  "features": [0.5, 1.2, 0.8, ...]
}
```

### POST /api/predict_batch
Upload CSV for batch prediction
- Form data with 'file' field
- Returns predictions for all samples

### GET /api/statistics
Get detection statistics
```json
{
  "total_detections": 150,
  "attack_count": 45,
  "benign_count": 105,
  "attack_types": {...}
}
```

### GET /api/model_info
Get model information
```json
{
  "model_type": "RandomForestClassifier",
  "features_count": 46,
  "attack_categories": [...]
}
```

### GET /api/clear_log
Clear detection logs

## 🛠️ Customization

### Using Different Algorithms

You can train with different algorithms by modifying `iot_attack_detector.py`:

```python
# Instead of Random Forest:
detector.train_random_forest(X_train, y_train)

# Try Gradient Boosting:
detector.train_gradient_boosting(X_train, y_train)
```

### Hyperparameter Tuning

Modify hyperparameters in the training methods:

```python
RandomForestClassifier(
    n_estimators=200,      # More trees
    max_depth=25,          # Deeper trees
    min_samples_split=10,  # More conservative splitting
    # ...
)
```

## 📊 Example Use Cases

### 1. Real-time Network Monitoring
Deploy in IoT network to monitor traffic and detect attacks in real-time

### 2. Security Auditing
Analyze historical network traffic logs to identify past attacks

### 3. Research & Development
Use as baseline for developing new attack detection algorithms

### 4. Education & Training
Teach cybersecurity concepts and ML applications

## ⚠️ Important Notes

1. **Model Retraining**: Retrain periodically with new attack patterns
2. **Feature Consistency**: Ensure prediction features match training features
3. **Class Imbalance**: Dataset may have imbalanced classes; consider using techniques like SMOTE
4. **False Positives**: Monitor and adjust threshold for production use
5. **Resource Usage**: Batch predictions require more memory

## 🤝 Contributing

Improvements welcome! Consider:
- Adding more ML algorithms (XGBoost, Neural Networks)
- Implementing real-time packet capture
- Adding alert notifications
- Creating mobile app interface
- Improving visualization

## 📚 References

- **Dataset**: Neto et al., "CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment"
- **Source**: University of New Brunswick - Canadian Institute for Cybersecurity
- **Website**: https://www.unb.ca/cic/datasets/iotdataset-2023.html

## 📝 License

This project uses the UNB CIC IoT Dataset 2023. Please cite the dataset in any publications:

```
Neto, E. C. P., Dadkhah, S., Ferreira, R., Zohourian, A., Lu, R., & Ghorbani, A. A. (2023). 
CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment. 
Sensors, 23(13), 5941.
```

## 🐛 Troubleshooting

### Model not loading
- Ensure you've trained and saved the model first
- Check that `iot_attack_detector.pkl` exists

### Feature mismatch error
- Verify input features match training data
- Check feature count (should be 46 or 47)

### Memory errors
- Reduce batch size
- Use smaller dataset sample for training
- Increase system RAM

### Web dashboard not accessible
- Check if port 5000 is available
- Try different port: `app.run(port=8080)`

## 📧 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the CIC dataset documentation
3. Examine the classification report after training

## 🎓 Learning Resources

- **Machine Learning**: scikit-learn documentation
- **IoT Security**: OWASP IoT Security Guide
- **Dataset Paper**: Read the full CICIoT2023 paper for methodology details

---

**Built with ❤️ for IoT Security Research**
