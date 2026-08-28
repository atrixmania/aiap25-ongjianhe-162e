# =========================
# src/config.py
# =========================

import os

print("Config for machine learning")


# =========================
# MODEL CONFIG
# =========================

DEFAULT_MODEL = "Logistic Regression"
ENV_MODEL = os.getenv("MODEL_TYPE")


# =========================
# PROJECT PATHS
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DATA_DIR = os.path.join(BASE_DIR, "data")

TRAINED_DIR = os.path.join(DATA_DIR, "trained")

PREDICTION_DIR = os.path.join(DATA_DIR, "prediction")


MODEL_DIR = os.path.join(TRAINED_DIR, "model")


# =========================
# CREATE REQUIRED FOLDERS
# =========================

REQUIRED_DIRS = [DATA_DIR, TRAINED_DIR, PREDICTION_DIR, MODEL_DIR]


for folder in REQUIRED_DIRS: os.makedirs(folder, exist_ok=True)


CONFIG = {

    # =========================
    # EXISTING DATA LOADER CONFIG
    # KEEP
    # =========================

    "db_dir": DATA_DIR,

    "db_path": None,

    "multi_db": True,

    "join_key": "client_email",



    # =========================
    # DATABASE DISCOVERY
    # =========================

    # Search all database files here
    "db_extension": ".db",


    # Prediction folder
    "prediction_dir": PREDICTION_DIR,



    # =========================
    # TRAIN DATA
    # =========================

    "trained_dir":
        TRAINED_DIR,


    "train_data":
        os.path.join(TRAINED_DIR, "train.parquet"),


    # =========================
    # MODEL ARTIFACTS
    # =========================


    "preprocess_model":
        os.path.join(TRAINED_DIR, "preprocess.pkl"),


    "feature_model":
        os.path.join(TRAINED_DIR, "feature.pkl"),


    "service_model":
        os.path.join(TRAINED_DIR, "service_model.pkl"),


    "metadata_path": os.path.join(TRAINED_DIR, "training_metadata.json"),


    # =========================
    # TRAINING INFORMATION
    # =========================


    # Store last training information
    #"training_metadata":
    #    os.path.join(TRAINED_DIR, "training_metadata.json"),



    # =========================
    # MODEL SETTINGS
    # =========================

    "model_type":
        ENV_MODEL if ENV_MODEL else DEFAULT_MODEL,

    "test_size": 0.2,

    "random_state": 42,



    # =========================
    # MODEL PARAMETERS
    # =========================

    "rf_params":
        {"n_estimators": 200, "max_depth": None},

    "logistic_params":
        {"max_iter": 1000},

    "gb_params":
        {"n_estimators": 100, "learning_rate": 0.1}
}
