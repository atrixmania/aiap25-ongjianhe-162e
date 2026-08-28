# =========================================================
# src/app.py
# =========================================================

import os
import sys
import joblib
import pandas as pd

from datetime import datetime
from threading import Timer
import webbrowser

from sklearn.model_selection import train_test_split


# =========================================================
# PROJECT BASE DIRECTORY
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# =========================================================
# LOG DIRECTORY
# =========================================================

LOG_DIR = os.path.join(
    BASE_DIR,
    "data",
    "logs"
)

os.makedirs(
    LOG_DIR,
    exist_ok=True
)


# =========================================================
# LOG FILE
# =========================================================

timestamp = datetime.now().strftime(
    "%Y-%m-%d_%H-%M-%S"
)

LOG_FILE = os.path.join(
    LOG_DIR,
    f"log_{timestamp}.log"
)


# =========================================================
# TEE OUTPUT
# =========================================================

class Tee:

    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):

        for stream in self.streams:

            try:
                stream.write(message)
                stream.flush()

            except Exception:
                pass

    def flush(self):

        for stream in self.streams:

            try:
                stream.flush()

            except Exception:
                pass

    def isatty(self):
        return False


# =========================================================
# START LOGGING
# =========================================================

log_file = open(
    LOG_FILE,
    "a",
    encoding="utf-8"
)

original_stdout = sys.stdout
original_stderr = sys.stderr

sys.stdout = Tee(
    original_stdout,
    log_file
)

sys.stderr = Tee(
    original_stderr,
    log_file
)


print("=" * 80)
print("AI SG ASSESSMENT - APPLICATION START")
print("=" * 80)
print(f"BASE_DIR : {BASE_DIR}")
print(f"LOG_FILE : {LOG_FILE}")
print("=" * 80)


# =========================================================
# PROJECT IMPORTS
# =========================================================

from config import CONFIG

from data_loader import (
    load_data,
    get_latest_db,
    need_retrain,
    save_training_metadata,
    load_single_db,
    remove_duplicates
)

from preprocess import DataProcessor
from feature_engineering import FeatureEngineer
from model import ServiceModel

from dash import Dash, html, dcc
from dash.dependencies import Input, Output


print("[INFO] Loading application modules...")


# =========================================================
# GLOBAL ML OBJECTS
# =========================================================

processor = None
feature_engineer = None
service_model = None


# =========================================================
# GLOBAL DATA
# =========================================================

# Dashboard dataframe
df = None

# Prediction dataframe
prediction_df = None


# =========================================================
# DASHBOARD DATABASE SELECTION
# =========================================================

def get_dashboard_db():

    """
    Select database for dashboard.

    Priority:

        1. data/prediction/*.db
        2. data/*.db

    The prediction database is selected only once.

    It is NOT loaded again through
    load_prediction_data().
    """

    print(
        "[INFO] Searching for dashboard database..."
    )

    data_dir = CONFIG["db_dir"]

    prediction_dir = CONFIG.get(
        "prediction_dir"
    )

    db_extension = CONFIG.get(
        "db_extension",
        ".db"
    )

    # =====================================================
    # 1. PREDICTION DATABASE
    # =====================================================

    prediction_dbs = []

    if (
        prediction_dir
        and
        os.path.isdir(prediction_dir)
    ):

        prediction_dbs = [
            os.path.join(
                prediction_dir,
                filename
            )
            for filename in os.listdir(
                prediction_dir
            )
            if filename.lower().endswith(
                db_extension.lower()
            )
        ]

    if prediction_dbs:

        prediction_dbs.sort(
            key=os.path.getmtime,
            reverse=True
        )

        selected_db = prediction_dbs[0]

        print(
            "[INFO] Prediction database found."
        )

        print(
            f"[INFO] Prediction DB count: "
            f"{len(prediction_dbs)}"
        )

        print(
            f"[INFO] Dashboard DB source: "
            f"{selected_db}"
        )

        print(
            f"[INFO] DB modified: "
            f"{datetime.fromtimestamp(os.path.getmtime(selected_db))}"
        )

        return selected_db

    # =====================================================
    # 2. MAIN DATABASE FALLBACK
    # =====================================================

    main_dbs = []

    if os.path.isdir(data_dir):

        main_dbs = [
            os.path.join(
                data_dir,
                filename
            )
            for filename in os.listdir(
                data_dir
            )
            if filename.lower().endswith(
                db_extension.lower()
            )
        ]

    if main_dbs:

        main_dbs.sort(
            key=os.path.getmtime,
            reverse=True
        )

        selected_db = main_dbs[0]

        print(
            "[INFO] No prediction database found."
        )

        print(
            "[INFO] Falling back to main database."
        )

        print(
            f"[INFO] Main DB count: "
            f"{len(main_dbs)}"
        )

        print(
            f"[INFO] Dashboard DB source: "
            f"{selected_db}"
        )

        print(
            f"[INFO] DB modified: "
            f"{datetime.fromtimestamp(os.path.getmtime(selected_db))}"
        )

        return selected_db

    raise FileNotFoundError(
        "No dashboard database found.\n"
        f"Prediction directory: {prediction_dir}\n"
        f"Main data directory: {data_dir}"
    )


# =========================================================
# LOAD RAW DATABASE FOR DASHBOARD
# =========================================================

def load_dashboard_raw_data(db_path):

    """
    Load dashboard database exactly once.

    The selected database is loaded directly.

    source_db is added because DataProcessor expects
    this column during preprocessing.
    """

    print(
        "[INFO] Explicit database path supplied."
    )

    print(
        "[INFO] Bypassing saved training dataset."
    )

    print(
        f"[INFO] Raw database selected: {db_path}"
    )

    if not db_path or not os.path.exists(db_path):

        raise FileNotFoundError(
            f"Dashboard database not found: {db_path}"
        )

    join_key = CONFIG.get(
        "join_key",
        "delivery_id"
    )

    # -----------------------------------------------------
    # LOAD DATABASE
    # -----------------------------------------------------

    raw_df = load_single_db(
        db_path,
        join_key
    )

    # -----------------------------------------------------
    # ADD SOURCE DATABASE
    #
    # IMPORTANT:
    # DataProcessor expects source_db.
    # load_multiple_dbs() normally adds this column,
    # but load_single_db() does not.
    # -----------------------------------------------------

    if "source_db" not in raw_df.columns:

        raw_df["source_db"] = os.path.basename(
            db_path
        )

    # -----------------------------------------------------
    # REMOVE DUPLICATES
    # -----------------------------------------------------

    raw_df = remove_duplicates(
        raw_df
    )

    print(
        f"[INFO] Raw dashboard data shape: "
        f"{raw_df.shape}"
    )

    return raw_df

# =========================================================
# TRAINING
# =========================================================

def initialize_training():

    global processor
    global feature_engineer
    global service_model
    global df

    print()
    print("=" * 40)
    print("TRAINING CHECK")
    print("=" * 40)

    # =====================================================
    # ARTIFACT PATHS
    # =====================================================

    preprocess_path = CONFIG[
        "preprocess_model"
    ]

    feature_path = CONFIG[
        "feature_model"
    ]

    service_model_path = CONFIG[
        "service_model"
    ]

    train_data_path = CONFIG.get(
        "train_data"
    )

    # =====================================================
    # CHECK ARTIFACTS
    # =====================================================

    artifacts_exist = (
        os.path.exists(preprocess_path)
        and
        os.path.exists(feature_path)
        and
        os.path.exists(service_model_path)
    )

    # =====================================================
    # GET TRAINING DB
    # =====================================================

    latest_db = get_latest_db()

    retrain_required = need_retrain(
        latest_db
    )

    # =====================================================
    # TRAINING REQUIRED
    # =====================================================

    if (
        not artifacts_exist
        or
        retrain_required
    ):

        print(
            "[INFO] Training required."
        )

        # -------------------------------------------------
        # LOAD TRAINING DATA
        # -------------------------------------------------

        print(
            "[INFO] Loading training data..."
        )

        raw_training_df = load_data(
            CONFIG
        )

        print(
            f"[INFO] Training data shape: "
            f"{raw_training_df.shape}"
        )

        # -------------------------------------------------
        # Preparing final_rating target
        # -------------------------------------------------

        print(
            "[INFO] Preparing final_rating target..."
        )

        processor = DataProcessor(
            CONFIG
        )

        raw_training_df = processor.prepare_final_rating(
            raw_training_df
        )

        print(
            "[INFO] final_rating created."
        )

        print(
            "[INFO] final_rating distribution:"
        )

        print(
            raw_training_df["final_rating"]
            .value_counts()
            .sort_index()
        )
                


        # =================================================
        # 2. SPLIT RAW DATA
        # =================================================

        train_df, temp_df = train_test_split(
            raw_training_df,
            test_size=0.20,
            random_state=42,
            stratify=raw_training_df["final_rating"]
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.50,
            random_state=42,
            stratify=temp_df["final_rating"]
        )

        print(
            f"[INFO] Train rows      : {len(train_df)}"
        )

        print(
            f"[INFO] Validation rows : {len(val_df)}"
        )

        print(
            f"[INFO] Test rows       : {len(test_df)}"
        )      

        # -------------------------------------------------
        # DATA PROCESSOR
        # -------------------------------------------------

        #print(
        #    "[INFO] Running DataProcessor..."
        #)

        #processor = DataProcessor(
        #    CONFIG
        #)


        # -------------------------------------------------
        # FIT ONLY ON TRAINING DATA
        # -------------------------------------------------

        print(
            "[INFO] Fitting DataProcessor on TRAIN only..."
        )

        processor.fit(train_df)
                

        train_df = processor.fit_transform(
            train_df
        )


        # -------------------------------------------------
        # TRANSFORM VALIDATION
        # -------------------------------------------------

        print(
            "[INFO] Transforming validation data..."
        )

        val_df = processor.transform(
            val_df
        )


        # -------------------------------------------------
        # TRANSFORM TEST
        # -------------------------------------------------

        print(
            "[INFO] Transforming test data..."
        )

        test_df = processor.transform(
            test_df
        )




        # -------------------------------------------------
        # SAVE PREPROCESSOR
        # -------------------------------------------------

        joblib.dump(
            processor,
            preprocess_path
        )

        print(
            f"[INFO] Preprocessor saved: "
            f"{preprocess_path}"
        )

        # -------------------------------------------------
        # FEATURE ENGINEERING
        # -------------------------------------------------

        print(
            "[INFO] Running FeatureEngineer..."
        )

        feature_engineer = FeatureEngineer(
            CONFIG
        )





        # -------------------------------------------------
        # FIT ONLY ON TRAINING DATA
        # -------------------------------------------------

        print(
            "[INFO] Fitting FeatureEngineer on TRAIN only..."
        )

        train_df = feature_engineer.fit_transform(
            train_df
        )


        # -------------------------------------------------
        # TRANSFORM VALIDATION
        # -------------------------------------------------

        print(
            "[INFO] Transforming validation features..."
        )

        val_df = feature_engineer.transform(
            val_df
        )


        # -------------------------------------------------
        # TRANSFORM TEST
        # -------------------------------------------------

        print(
            "[INFO] Transforming test features..."
        )

        test_df = feature_engineer.transform(
            test_df
        )


        # -------------------------------------------------
        # SAVE FEATURE ENGINEER
        # -------------------------------------------------

        joblib.dump(
            feature_engineer,
            feature_path
        )

        print(
            f"[INFO] Feature engineer saved: "
            f"{feature_path}"
        )

        # -------------------------------------------------
        # TRAIN SERVICE MODEL
        # -------------------------------------------------

        print(
            "[INFO] Training ServiceModel..."
        )

        service_model = ServiceModel(
            CONFIG
        )

        service_model.train(
            train_df,
            val_df,
            test_df
        )

        print(
            f"[INFO] Service model saved: "
            f"{service_model_path}"
        )

        # -------------------------------------------------
        # SAVE TRAINING METADATA
        # -------------------------------------------------

        save_training_metadata(
            latest_db
        )

        print(
            "[INFO] Training completed."
        )

    # =====================================================
    # NO TRAINING REQUIRED
    # =====================================================

    else:

        print(
            "[INFO] Existing trained artifacts found."
        )

        print(
            "[INFO] Training data unchanged."
        )

        print(
            "[INFO] Loading trained artifacts..."
        )

        # -------------------------------------------------
        # LOAD PREPROCESSOR
        # -------------------------------------------------

        processor = joblib.load(
            preprocess_path
        )

        print(
            "[INFO] Preprocessor loaded."
        )

        # -------------------------------------------------
        # LOAD FEATURE ENGINEER
        # -------------------------------------------------

        feature_engineer = joblib.load(
            feature_path
        )

        print(
            "[INFO] Feature engineer loaded."
        )

        # -------------------------------------------------
        # LOAD SERVICE MODEL
        # -------------------------------------------------

        service_model = joblib.load(
            service_model_path
        )

        print(
            "[INFO] Service model loaded."
        )

        # =================================================
        # IMPORTANT
        #
        # DO NOT LOAD train.parquet INTO df HERE.
        #
        # The dashboard must use the CURRENT dashboard DB.
        #
        # train.parquet is a processed training artifact.
        # It must NOT be passed through processor.transform()
        # again.
        # =================================================

        print(
            "[INFO] Selecting dashboard database..."
        )

        dashboard_db = get_dashboard_db()

        print(
            "[INFO] Loading raw dashboard data..."
        )

        df = load_dashboard_raw_data(
            dashboard_db
        )

        # -------------------------------------------------
        # PREPROCESS DASHBOARD DATA
        # -------------------------------------------------

        print(
            "[INFO] Processing dashboard data..."
        )

        df = processor.transform(
            df
        )

        print(
            "[INFO] DataProcessor transform completed"
        )

        # -------------------------------------------------
        # FEATURE ENGINEERING
        # -------------------------------------------------

        print(
            "[INFO] Applying feature engineering..."
        )

        df = feature_engineer.transform(
            df
        )

        print(
            "[INFO] FeatureEngineer transform completed"
        )

        print(
            f"[INFO] Processed dashboard data shape: "
            f"{df.shape}"
        )


    # =====================================================
    # LOAD CURRENT DASHBOARD DATA AFTER TRAINING
    # =====================================================

    if df is None:

        print(
            "[INFO] Loading dashboard data after training..."
        )

        dashboard_db = get_dashboard_db()

        print(
            "[INFO] Loading raw dashboard data..."
        )

        df = load_dashboard_raw_data(
            dashboard_db
        )

        print(
            "[INFO] Processing dashboard data..."
        )

        df = processor.transform(
            df
        )

        print(
            "[INFO] DataProcessor transform completed"
        )

        print(
            "[INFO] Applying feature engineering..."
        )

        df = feature_engineer.transform(
            df
        )

        print(
            "[INFO] FeatureEngineer transform completed"
        )

        print(
            f"[INFO] Processed dashboard data shape: "
            f"{df.shape}"
        )


        

    # =====================================================
    # FINAL CHECK
    # =====================================================

    if df is None:

        raise RuntimeError(
            "Dashboard dataframe was not initialized."
        )

    print(
        f"[INFO] Final dashboard dataframe: "
        f"{df.shape}"
    )

    print(
        "[INFO] Training pipeline ready."
    )


# =========================================================
# PREDICTION
# =========================================================

def initialize_prediction():

    global prediction_df

    print()
    print("=" * 40)
    print("PREDICTION CHECK")
    print("=" * 40)

    # =====================================================
    # IMPORTANT
    #
    # DO NOT LOAD prediction DB AGAIN.
    #
    # get_dashboard_db() already selected the prediction
    # database and initialize_training() already loaded it.
    #
    # Therefore prediction uses the SAME processed
    # dashboard dataframe.
    # =====================================================

    prediction_dir = CONFIG.get(
        "prediction_dir"
    )

    if not prediction_dir:

        print(
            "[INFO] Prediction directory not configured."
        )

        return

    prediction_dbs = [
        os.path.join(
            prediction_dir,
            filename
        )
        for filename in os.listdir(
            prediction_dir
        )
        if filename.lower().endswith(".db")
    ]

    if not prediction_dbs:

        print(
            "[INFO] No prediction databases found."
        )

        print(
            "[INFO] Prediction skipped."
        )

        return

    print(
        f"[INFO] Prediction DB count: "
        f"{len(prediction_dbs)}"
    )

    # =====================================================
    # USE ALREADY PROCESSED DASHBOARD DATA
    # =====================================================

    prediction_df = df.copy()

    print(
        f"[INFO] Prediction data using existing "
        f"dashboard dataframe: {prediction_df.shape}"
    )

    # =====================================================
    # LOAD SAVED SERVICE MODEL
    # =====================================================

    service_model_path = CONFIG[
        "service_model"
    ]

    if not os.path.exists(
        service_model_path
    ):

        raise FileNotFoundError(
            f"Service model not found: "
            f"{service_model_path}"
        )

    print(
        f"[INFO] Using trained service model: "
        f"{service_model_path}"
    )

    # -----------------------------------------------------
    # Make sure the service model is the saved artifact.
    # -----------------------------------------------------

    saved_service_model = joblib.load(
        service_model_path
    )

    print(
        "[INFO] service_model.pkl loaded."
    )

    # =====================================================
    # FAST BATCH PREDICTION
    # =====================================================

    print(
        "[INFO] Running model prediction..."
    )

    # =====================================================
    # BUILD SERVICE RATING FEATURES
    # =====================================================

    structured_features = [
        "comment",
        "hotel",
        "meal",
        "market_segment",
        "distribution_channel",
        "deposit_type",
        "customer_type",
        "season_group",
        "country_group",
        "adr_group",
        "previous_cancellations_group",
        "days_in_waiting_list_group",
        "guest_group",
        "client_segment",
        "is_top_bad_client",
        "client_bad_rate"
    ]

    input_df = prediction_df.copy()

    # -----------------------------------------------------
    # Ensure required columns exist
    # -----------------------------------------------------

    for col in structured_features:

        if col not in input_df.columns:

            input_df[col] = "unknown"

    X_input = input_df[
        structured_features
    ]

    # =====================================================
    # SERVICE RATING PREDICTION
    # =====================================================

    best_model = (
        saved_service_model.best_model
    )

    pred_ratings = best_model.predict(
        X_input
    )

    # =====================================================
    # PROBABILITY / DECISION SCORES
    # =====================================================

    clf = best_model.named_steps["clf"]

    if hasattr(
        clf,
        "predict_proba"
    ):

        probs = best_model.predict_proba(
            X_input
        )

    else:

        scores = best_model.decision_function(
            X_input
        )

        scores = pd.DataFrame(
            scores
        ).to_numpy()

        if scores.ndim == 1:

            import numpy as np

            scores = np.column_stack(
                [
                    -scores,
                    scores
                ]
            )

        import numpy as np

        scores = (
            scores -
            np.max(
                scores,
                axis=1,
                keepdims=True
            )
        )

        exp_scores = np.exp(
            scores
        )

        probs = (
            exp_scores /
            exp_scores.sum(
                axis=1,
                keepdims=True
            )
        )

    # =====================================================
    # RISK CALCULATION
    # =====================================================

    import numpy as np

    class_order = list(
        clf.classes_
    )

    risk_weights = {
        0: 1.0,
        1: 0.90,
        2: 0.75,
        3: 0.60,
        4: 0.40,
        5: 0.20,
        6: 0.00,
    }

    sla_risk = np.zeros(
        len(pred_ratings),
        dtype=float
    )

    for i, cls in enumerate(
        class_order
    ):

        if i >= probs.shape[1]:
            continue

        sla_risk += (
            probs[:, i] *
            risk_weights.get(
                cls,
                0
            )
        )

    sla_risk_score = (
        sla_risk * 100
    )

    # =====================================================
    # BASE RISK
    # =====================================================

    rating_map = {
        0: 100,
        1: 90,
        2: 75,
        3: 60,
        4: 40,
        5: 20,
        6: 10
    }

    base_risk = np.array(
        [
            rating_map.get(
                int(rating),
                50
            )
            for rating in pred_ratings
        ],
        dtype=float
    )

    risk_score = (
        0.6 * base_risk +
        0.4 * sla_risk_score
    )

    # =====================================================
    # RESERVATION STATUS PREDICTION
    # =====================================================
    #
    # Uses the reservation-status model already stored
    # inside ServiceModel.
    #
    # This is BATCH prediction, so there is no:
    #
    # for _, row in prediction_df.iterrows()
    #
    # =====================================================

    print(
        "[INFO] Running reservation status prediction..."
    )

    # -----------------------------------------------------
    # STATUS PREPROCESSING
    # -----------------------------------------------------

    X_status = (
        saved_service_model.status_preprocess.transform(
            X_input
        )
    )

    # -----------------------------------------------------
    # STATUS PROBABILITIES
    # -----------------------------------------------------

    status_probs = (
        saved_service_model.status_model.predict_proba(
            X_status
        )
    )

    status_class_order = (
        saved_service_model.status_model.classes_
    )

    # -----------------------------------------------------
    # Create arrays for each reservation status
    # -----------------------------------------------------

    prob_canceled = np.zeros(
        len(prediction_df),
        dtype=float
    )

    prob_check_out = np.zeros(
        len(prediction_df),
        dtype=float
    )

    prob_no_show = np.zeros(
        len(prediction_df),
        dtype=float
    )

    # -----------------------------------------------------
    # Match model classes to probability columns
    # -----------------------------------------------------

    for i, cls in enumerate(
        status_class_order
    ):

        probability = (
            status_probs[:, i] * 100
        )

        if cls == "canceled":

            prob_canceled = probability

        elif cls == "check-out":

            prob_check_out = probability

        elif cls == "no-show":

            prob_no_show = probability

    # =====================================================
    # ADD PREDICTIONS
    # =====================================================

    prediction_df[
        "predicted_rating"
    ] = pred_ratings

    prediction_df[
        "sla_risk_score"
    ] = np.round(
        sla_risk_score,
        1
    )

    prediction_df[
        "risk_score"
    ] = np.round(
        risk_score,
        1
    )

    # =====================================================
    # ADD RESERVATION STATUS PROBABILITIES
    # =====================================================

    prediction_df[
        "prob_canceled"
    ] = np.round(
        prob_canceled,
        1
    )

    prediction_df[
        "prob_check_out"
    ] = np.round(
        prob_check_out,
        1
    )

    prediction_df[
        "prob_no_show"
    ] = np.round(
        prob_no_show,
        1
    )

    # -----------------------------------------------------
    # Determine most likely reservation status
    # -----------------------------------------------------

    status_probability_matrix = np.column_stack(
        [
            prob_canceled,
            prob_check_out,
            prob_no_show
        ]
    )

    status_names = np.array(
        [
            "canceled",
            "check-out",
            "no-show"
        ]
    )

    prediction_df[
        "predicted_reservation_status"
    ] = status_names[
        np.argmax(
            status_probability_matrix,
            axis=1
        )
    ]

    print(
        "[INFO] Reservation status prediction completed."
    )

    # =====================================================
    # GENERATED COMMENT
    # =====================================================

    # -----------------------------------------------------
    # IMPORTANT
    #
    # Do NOT call _generate_comment() 150,000 times.
    #
    # Generate comments only when the UI needs them.
    # -----------------------------------------------------

    prediction_df[
        "generated_comment"
    ] = ""

    # =====================================================
    # FINAL INFORMATION
    # =====================================================

    print(
        f"[INFO] Prediction completed: "
        f"{len(prediction_df)} records"
    )

    print(
        "[INFO] Batch prediction completed."
    )

    print(
        "========================================"
    )


# =========================================================
# INITIALIZE ML PIPELINE
# =========================================================

initialize_training()


# =========================================================
# INITIALIZE PREDICTION
# =========================================================

initialize_prediction()


# =========================================================
# IMPORT DASH PAGES
# =========================================================

import eda_page
import predict_page


# =========================================================
# CREATE DASH APPLICATION
# =========================================================

app = Dash(
    __name__,
    suppress_callback_exceptions=True
)


# =========================================================
# REGISTER CALLBACKS
# =========================================================

print(
    "[INFO] Registering EDA callbacks..."
)

eda_page.register_callbacks(
    app,
    df
)

print(
    "[INFO] Registering Prediction callbacks..."
)

predict_page.register_callbacks(
    app,
    df
)


# =========================================================
# APP LAYOUT
# =========================================================

app.layout = html.Div([

    dcc.Location(
        id="url",
        refresh=False
    ),

    html.Div(
        id="page-content"
    )
])


# =========================================================
# ROUTING
# =========================================================

@app.callback(
    Output(
        "page-content",
        "children"
    ),
    Input(
        "url",
        "pathname"
    )
)
def display_page(pathname):

    if pathname == "/eda":

        return eda_page.create_layout(
            df
        )

    elif pathname == "/predict":

        return predict_page.create_layout(
            df
        )

    else:

        return html.Div([

            html.H1(
                "Main Dashboard",
                style={
                    "textAlign": "center"
                }
            ),

            html.Div([

                dcc.Link(
                    html.Button(
                        "Go to EDA Dashboard",
                        style={
                            "fontSize": "20px"
                        }
                    ),
                    href="/eda"
                ),

                dcc.Link(
                    html.Button(
                        "Go to Predict Dashboard",
                        style={
                            "fontSize": "20px"
                        }
                    ),
                    href="/predict"
                )

            ],
            style={
                "display": "flex",
                "justifyContent": "center",
                "gap": "20px",
                "marginTop": "50px"
            })

        ])


# =========================================================
# AUTO OPEN BROWSER
# =========================================================

def open_browser():

    webbrowser.open_new(
        "http://127.0.0.1:8050/"
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    Timer(
        1,
        open_browser
    ).start()

    app.run(
        debug=True,
        port=8050,
        use_reloader=False
    )
