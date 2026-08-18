
# Session-Based Psychometric Risk Detection using LSTM with SMOTE-Enhanced Class Balancing

This project identifies behavioral and psychometric risk factors from sequential user sessions by combining recurrent deep learning models with data augmentation techniques to solve severe class imbalance issues.

##  Project Overview
Psychometric risk assessment from active user sessions typically suffers from heavily skewed data distributions, as high-risk anomalies are rare compared to standard user behavior. This framework leverages Long Short-Term Memory (LSTM) networks to capture vital temporal and sequential dependencies across time-series sessions. To prevent model bias toward majority classes, the Synthetic Minority Over-sampling Technique (SMOTE) is integrated into the pre-processing pipeline, ensuring balanced training dynamics and reliable high-risk detection.

##  Key Features
* **Sequential Risk Tracking:** Utilizes recurrent architectures to evaluate behavioral evolution across consecutive user sessions.
* **Overcoming Class Imbalance:** Employs SMOTE-driven data synthesis to expand minority-class representations without duplicating data points.
* **Deep Temporal Modeling:** Implements an LSTM-based network capable of remembering long-term dependencies in psychometric metrics.
* **Robust Evaluation Metrics:** Assesses performance using Precision, Recall, and F1-Score to confirm accuracy across rare risk boundaries.

##  Built With
* **Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras (or PyTorch depending on environment execution)
* **Data Augmentation:** Imbalanced-learn (`imblearn` for SMOTE implementation)
* **Data Analytics:** Scikit-learn, Pandas, NumPy, Matplotlib

## Repository Structure
```text
└── Session_Based_psychometric_Risk_Detection_using_LSTM_with_SMOTE_enhanced_class_balancing.ipynb  
    # Complete end-to-end notebook detailing data ingestion, SMOTE scaling, LSTM layers, training logs, and performance validation graphs.
```

##  Setup & Execution

### Prerequisites
Install the required packages to handle deep learning and imbalanced datasets:
```bash
pip install numpy pandas scikit-learn tensorflow imbalanced-learn matplotlib
```

### Running the Notebook
1. Clone this repository:
   ```bash
   git clone https://github.com
   cd Session-Based-psychometric-Risk-Detection-using-LSTM-with-SMOTE-enhanced-class-balancing
   ```
2. Launch Jupyter Notebook or upload the file to Google Colab.
3. Execute the data preparation and SMOTE augmentation blocks before training the LSTM network layers.
