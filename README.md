# 📊 FinDS: Feature-Driven Financial Time Series Forecasting Framework

## 🚀 Overview

**FinDS (Financial Data Science)** is a research-oriented framework for studying the impact of feature selection and feature behavior on financial time-series forecasting.

The project introduces a structured experimental pipeline to evaluate how different feature subsets influence model performance and generalization.

---

## 🎯 Objectives

- Analyze the role of features in financial forecasting  
- Identify relevant vs redundant features  
- Evaluate performance under different feature subsets  
- Study feature behavior and stability  

---

## 🏗️ Repository Structure

FinDS/
│
├── 1. Main Exp/
│   ├── Baseline forecasting experiments
│   ├── Training with full feature set
│   └── Benchmark results
│
├── 2. Static_Features_selection/
│   ├── Feature selection methods
│   ├── Feature ranking / filtering
│   └── Reduced feature experiments
│
├── 3. FBGFS_Analysis/
│   ├── Feature behavior analysis
│   ├── Feature importance evaluation
│   └── Comparative studies
│
└── README.md

---

## ⚙️ Experimental Pipeline

### 🔹 Stage 1: Baseline Experiment
- Train models using all available features  
- Establish baseline performance  

### 🔹 Stage 2: Static Feature Selection
- Apply feature selection techniques:
  - Correlation filtering  
  - Statistical ranking  
  - Model-based importance  

- Remove redundant and irrelevant features  
- Retrain models on reduced feature sets  

### 🔹 Stage 3: FBGFS Analysis
- Analyze feature importance and contribution  
- Study feature interaction and stability  
- Compare full vs reduced feature performance  

---

## 📊 Workflow

Raw Data
   ↓
Preprocessing
   ↓
Baseline Model (All Features)
   ↓
Feature Selection
   ↓
Reduced Model Training
   ↓
FBGFS Analysis
   ↓
Evaluation & Comparison

---

## 📈 Evaluation Metrics

- MAPE – Mean Absolute Percentage Error  
- NMSE – Normalized Mean Squared Error  
- NRMSE – Normalized Root Mean Squared Error  
- Directional Accuracy (DA) – Trend prediction accuracy  

---

## 🧠 Research Motivation

Financial datasets are:
- High-dimensional  
- Noisy  
- Redundant  

Using all features can lead to:
- Overfitting  
- Poor generalization  
- Low interpretability  

This framework addresses these issues through structured feature analysis.

---

## 🔬 Applications

- Financial time-series forecasting  
- Feature selection research  
- Model interpretability  
- Quantitative finance  

---

## 🛠️ Tech Stack

- Python  
- NumPy, Pandas  
- Scikit-learn  
- PyTorch  

---

## 🔮 Future Work

- Dynamic feature selection  
- Graph-based feature modeling  
- Causal feature discovery  
- Integration with Transformers and GNNs  

---

## 🤝 Contributing

git clone https://github.com/prof-manav/FinDS.git
cd FinDS

git checkout -b feature-name
git commit -m "Add feature"
git push origin feature-name

---

## 📄 License

MIT License

---

## 👨‍💻 Author

Manav  
AI Researcher | Time Series Forecasting
