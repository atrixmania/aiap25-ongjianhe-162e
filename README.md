## AIAP Batch 25 Technical Assessment - Exploratory Data Analysis (EDA)

**Candidate Name:** Ong Jian He
**Email:** faithwish@hotmail.com



\*\*\*IMPORTANT NOTE: Please put the Dataset (\*.db) in data folder for eda.ipynb and run.sh to work.



## 

## Project Overview



This project builds an end-to-end machine learning pipeline to predict Travelers’ Inn service level expectation and no show booking by customer.



The objective is to help provide insight on the service level expectation, customer no show booking and how likely it will run in trouble.





## Instructions for Executing and Modifying the Pipeline

### Prerequisites



Step 1 - Install Python version 3.14 (if not installed) from https://www.python.org/downloads/

Step 2 - install Git from https://git-scm.com/install/windows

Step 3 - right click "project" folder name and click "Open Git Bash here"

step 4 - type "bash run.sh" to run (it will automatically download hotel.db if file not available)



Note: not require to type "pip install -r requirements.txt" as it is coded in bash run.sh



Parameters can be changed in `config.py`.







## Folder Structure



```
aiap24-ongjianhe-162E/
│
├── data/
│   ├── Dataset (*.db)          # Historical data used for training
│   │
│   ├── logs/
│   │   ├── logs_yyyy_mm_dd_hr_mm_ss.text      # Logs the terminal screen data from running code
│   │
│   ├── trained/
│   │   ├── train.parquet      # Preprocessed + feature engineered training data
│   │   ├── preprocess.pkl     # Fitted DataProcessor
│   │   ├── feature.pkl        # Fitted FeatureEngineer
│   │   └── service_model.pkl  # Trained ML model
│   │
│   │   └── training_metadata.json
│   │
│   └── prediction/
│       └── new_prediction dataset (*.db)  # New records for prediction
│
└── src/
│    ├── config.py
│    ├── app.py
│    ├── data_loader.py
│    ├── preprocess.py
│    ├── feature_engineering.py
│    ├── model.py
│    ├── eda_page.py
│    ├── predict_page.py
│
├── decision_log.md
├── eda.ipynb # EDA notebook
├── prompt_chat_history.md
├── readme.md
├── requirements.txt # Dependencies
└── run.sh # Execution script    
```

## 

## Logical Steps

```mermaid
Flowchart TD

    A[app.py]

    %% =========================
    %% TRAINING CHECK
    %% =========================

    A --> TC{Training required?}

    TC -->|YES| D[data_loader.py]
    TC -->|NO| L[Load Saved Artifacts]

    D --> P[DataProcessor<br/>fit_transform()]
    P --> F[FeatureEngineer<br/>fit_transform()]
    F --> M[ServiceModel<br/>train()]

    M --> SA[Save service_model.pkl]
    P --> SP[Save preprocess_model.pkl]
    F --> SF[Save feature_model.pkl]

    SA --> ART[(Saved ML Artifacts)]
    SP --> ART
    SF --> ART

    L --> ART

    %% =========================
    %% PREDICTION CHECK
    %% =========================

    A --> PC{Prediction DB available?}

    PC -->|YES| PD[data_loader.py<br/>load_prediction_data()]
    PC -->|NO| DASH[Start Dash]

    PD --> PT[DataProcessor<br/>transform()]
    PT --> FT[FeatureEngineer<br/>transform()]
    FT --> MP[ServiceModel<br/>predict()]
    MP --> PR[(Prediction Results)]

    %% =========================
    %% PRESENTATION
    %% =========================

    ART --> E[eda_page.py]
    ART --> R[predict_page.py]

    PR --> R

    E --> DASH
    R --> DASH

    %% =========================
    %% DASH
    %% =========================

    DASH --> UI[Dash Application<br/>/eda and /predict]


## 

## Pipeline Flow

1 Data Extraction (SQLite)
2 Data Cleaning (remove duplicate, convert to small text, remove space)
3 Feature Engineering
4 Preprocessing (encoding + imputation)
5 Model Training (3 models)
6 Evaluation & Model Selection
7 Prediction & Insight

```mermaid
Pipeline TD
    A["app.py"]

    %% =========================
    %% TRAINING
    %% =========================

    A --> B["initialize_training()"]

    B --> C["Training DB Check"]

    C --> D{"DB changed<br/>or artifacts missing?"}

    D -->|YES| E["Load data/*.db"]
    E --> F["processor.fit_transform()"]
    F --> G["feature_engineer.fit_transform()"]
    G --> H["service_model.train()"]
    H --> I["Save Model Artifacts"]

    D -->|NO| J["Load Existing<br/>Model Artifacts"]

    I --> K["Training Pipeline Ready"]
    J --> K

    %% =========================
    %% PREDICTION
    %% =========================

    A --> L["initialize_prediction()"]

    L --> M["Check data/prediction/"]

    M --> N{"Prediction DB exists?"}

    N -->|YES| O["Load prediction/*.db"]
    N -->|NO| P["No Prediction Data"]

    O --> Q["processor.transform()"]
    Q --> R["feature_engineer.transform()"]
    R --> S["service_model.predict()"]
    S --> T["Prediction Results"]

    %% =========================
    %% DASHBOARD
    %% =========================

    K --> U["START DASHBOARD"]
    P --> U
    T --> U

    U --> V["get_dashboard_db()"]

    V --> W{"Any *.db in<br/>data/prediction/?"}

    W -->|YES| X["PRIORITY SOURCE<br/>data/prediction/*.db"]

    W -->|NO| Y["FALLBACK SOURCE<br/>data/*.db"]

    X --> Z["dashboard_db"]
    Y --> Z

    Z --> AA["load_data(<br/>CONFIG,<br/>db_path=dashboard_db<br/>)"]

    AA --> AB["Load RAW DB"]

    AB --> AC["processor.transform()"]

    AC --> AD["feature_engineer.transform()"]

    AD --> AE["Dashboard DataFrame<br/>df"]

    AE --> AF["/eda"]
    AE --> AG["/predict"]


|TRAINING |PREDICTION|
|-|-|
|DataProcessor fit_transform()|DataProcessor transform()|
|FeatureEngineer fit_transform()|FeatureEngineer transform()|
|ServiceModel train()|ServiceModel predict()|
|save artifacts|use artifacts|




## 

## EDA Key Findings & Pipeline Choices



**Service Level:** Rating with less than 2 provide comment on the unfavourable service level 

**Reservation Level:** High percentage of no show and cancellation for customer who has make booking


These insights guided feature engineering and model selection.



Features show a consistent pattern where complexity, content richness, and modern design practices are all positively correlated with each other, while domain age shows an inverse relationship with content density/referencing.



**Class Balance:** Both booking and review classes are well-represented; no augmentation needed.


**Data Cleaning:**

  * Dropped duplicate row
  * change all word to small caps
  * Remove extra space
  * Fixed typos (e.g., 'no depost' → 'no deposit').
  * Converted negative values to positive.
  * Imputed missing values that using iterative imputer.



**Outliers:** Removed extreme outliers based on distribution.



**Feature Engineering:**

* Encoded categorical variables into bins.
* Binary encoding using sentence transformer for word / text
* Created new features (eg. customer behaviour)





## Feature Processing Summary

|Feature|Type|Processing Applied|
|-|-|-|
|missing data|numeric|ITERATIVE IMPUTER using machine learning|
|missing rating|numeric|ITERATIVE IMPUTER using machine learning|
|Data that has only word|word / text|binary encoding using sentence transformer|
|hotel, meal, market_segment, distribution_channel, deposit_type, customer_type, reservation_status|categorical|one-hot encoding|
|season_group, country_group, adr_group, previous_cancellations_group, days_in_waiting_list_group|categorical|Encoded categorical variables into bins|
|Client behaviour|categorical|one-hot encoding|
|Top comment|categorical|Encoded categorical variables into bins|

## 

## Model Choices



|Algorithm|Status|Rationale|
|-|-|-|
|Logistic Regression|Selected for evaluation|baseline, interpretable|
|LinearSVC|Selected for evaluation|Captures Maximum-Margin Hyperplanes|
|LGBMClassifier|Selected for evaluation|Captures Complex, Non-Linear Patterns which is alternative to Gradient Boosting|


## 

## Model Evaluation

Used **F1-score** and **ROC-AUC** metrics which are suitable for binary classification tasks. **F1-score** especially penalises false negatives/positives which is important for our use-case to minimise false detection of no show, cancellation and customer rating.



|Model|ROC-AUC Score|F1 Score|final score|
|-|-|-|-|
|Logistic Regression|0.790759|0.601019|0.714863|
|LinearSVC|0.790653|0.573403|0.703753|
|LGBMClassifier|0.790843|0.548502|0.693906|



All models achieved nearly identical ROC-AUC scores, but their classification balance varies significantly. We will first eliminate the **LGBMClassifier** because it delivers the weakest F1 Score, failing to capture minority class instances effectively despite its competitive ROC-AUC.

**LinearSVC** offers a slightly better F1 Score, but it lacks the native probabilistic outputs that are often crucial for setting flexible classification thresholds in production.

**Logistic Regression** emerges as the optimal choice for this deployment. It achieves the highest F1 Score by a clear margin while matching the peak ROC-AUC performance. Furthermore, it is computationally lightweight, exceptionally fast to train, highly interpretable, and carries a much lower risk of overfitting compared to the tree-based LightGBM model.

If Travelers’ Inn wants the most reliable, cost-efficient, and structurally balanced model, Logistic Regression is the best selection here.

