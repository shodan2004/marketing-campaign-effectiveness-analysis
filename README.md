



# Marketing Campaign Effectiveness Analysis

This project analyzes the effectiveness of marketing campaigns using customer data. It includes data preprocessing, feature engineering, one-hot encoding and normalization, machine learning model training with a `RandomForestClassifier`, and an interactive Streamlit dashboard for visualizing results and insights.

## Project Overview

* **Purpose**: Predict customer response to marketing campaigns and provide actionable insights based on feature importance and campaign performance metrics.
* **Dataset**: `marketing_campaign.xlsx`, processed into `marketing_campaign_ml_ready.xlsx` with engineered features and scaled/one-hot encoded variables.
* **Tools**: Python 3.9.6, pandas, scikit-learn, imbalanced-learn, seaborn, matplotlib, Streamlit.
* **Output**: Interactive dashboard to explore campaign effectiveness, response rates, and optimization suggestions.

## Project Structure

```
C:\Users\vemul\Downloads\marketing_campaign_analysis_new\
│
├── campaign_analysis.py                     # Streamlit application
├── requirements.txt                         # Python dependencies
├── One-hot encoding+normalization.ipynb     # Encoding & scaling notebook
├── pre_processing+feature_engineering.ipynb # Preprocessing & feature engineering notebook
├── main.ipynb                               # Model training & evaluation notebook
├── README.md                                # This file
└── data/                                    # Raw & processed data
    ├── raw/                                # Raw dataset (marketing_campaign.xlsx)
    └── processed/
        └── encoded/                        # ML-ready dataset (marketing_campaign_ml_ready.xlsx)
```

> **Note**: The `data` folder may need to be created manually. Adjust paths in scripts/notebooks if the data is elsewhere.

## Setup

### Prerequisites

* Python 3.9.6
* Required libraries (listed in `requirements.txt`)

### Installation

1. **Install Dependencies**:

```cmd
cd C:\Users\vemul\Downloads\marketing_campaign_analysis_new
pip install -r requirements.txt
```

2. **Prepare the Dataset**:

* Place `marketing_campaign.xlsx` in `data/raw/` (create folder if needed):

```cmd
mkdir data\raw
```

* Run preprocessing and encoding notebooks in Jupyter:

  * `pre_processing+feature_engineering.ipynb` → execute all cells
  * `One-hot encoding+normalization.ipynb` → execute all cells
* Save the final dataset as `marketing_campaign_ml_ready.xlsx` in `data/processed/encoded/`. Adjust `save_path` in notebooks if needed.

## Usage

1. **Run the Streamlit Dashboard**:

```cmd
cd C:\Users\vemul\Downloads\marketing_campaign_analysis_new
streamlit run campaign_analysis.py
```

2. **Open Dashboard**:

* Go to `http://localhost:8501` in your browser.
* Use sidebar filters to select education levels and marital statuses.
* Adjust the prediction threshold slider to see changes in model performance.
* Explore key performance indicators, response rates by category, feature importance, and campaign success insights.

## Features

* **Data Preprocessing**: Handles missing values, converts dates, creates features like `Age`, `Total_Kids`, `Customer_Tenure_Days`.
* **Feature Engineering**: Derived features like `Total_Spent`, `Avg_Spent_per_Purchase`, `High_Spender`.
* **Model Training**: RandomForestClassifier with optimized parameters (`n_estimators=223`, `max_depth=20`) and SMOTE for class imbalance.
* **Dashboard**:

  * Campaign response rate & class distribution
  * Response rates by education & marital status
  * Top feature importance & model metrics (accuracy, precision, recall, F1-score)
  * Insights & optimization suggestions based on feature importance
  * Analysis of previous campaign success impact

## Model Performance

* **Accuracy**: \~0.97
* **F1-Score (minority class)**: \~0.92
* **Best Parameters**: From RandomizedSearchCV in `main.ipynb`

## Troubleshooting

* **File Not Found**: Verify `marketing_campaign_ml_ready.xlsx` exists in `data/processed/encoded/`. Adjust path in `campaign_analysis.py` if necessary.
* **Dependencies**: Ensure all libraries are installed. Update pip if errors occur:

```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Contributing

1. Fork the repo
2. Create a branch (`git checkout -b feature-branch`)
3. Make changes & commit (`git commit -m "description"`)
4. Push (`git push origin feature-branch`)
5. Open a pull request

## Acknowledgments

* Dataset inspired by marketing campaign challenges
* Open-source libraries: pandas, scikit-learn, Streamlit

---

## Contact

**Shodhan**
📧 Email: [shodan.v3@gmail.com](mailto:shodan.v3@gmail.com)
🌐 Linked-in: \[https://www.linkedin.com/in/shodhan-vemulapalli]

