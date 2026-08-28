# =========================
# src/model.py
# =========================

import os
import joblib
import warnings
import numpy as np
import pandas as pd

from datetime import datetime

from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from lightgbm import LGBMClassifier

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import (
    HashingVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
    CountVectorizer
)

from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.decomposition import TruncatedSVD

from sklearn.exceptions import ConvergenceWarning


warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning
)


print("Running model.py")


# =========================================================
# COMMENT VALIDATION
# =========================================================

def is_valid_comment(c: str) -> bool:

    if c is None:
        return False

    c = str(c).strip()

    if len(c) < 5:
        return False

    if c.lower() in [
        "nan",
        "none",
        "null",
        ""
    ]:
        return False

    if c.count(" ") < 2:
        return False

    return True


# =========================================================
# SERVICE MODEL
# =========================================================

class ServiceModel:

    def __init__(self, config=None):

        self.config = config or {}

        # ---------------------------------------------
        # FINAL RATING MODEL
        # ---------------------------------------------

        self.best_model = None
        self.best_model_name = None

        self.models = {}
        self.results_df = None

        # ---------------------------------------------
        # REFERENCE DATA
        # ---------------------------------------------

        self.df_reference = None

        # ---------------------------------------------
        # COMMENT MODEL
        # ---------------------------------------------

        self.comment_df = None
        self.comment_index = {}

        self.comment_tfidf = None
        self.comment_svd = None

        # ---------------------------------------------
        # RESERVATION STATUS MODEL
        # ---------------------------------------------

        self.status_model = None
        self.status_preprocess = None

        self.status_features = []
        self.status_numeric_features = []
        self.status_categorical_features = []

        # ---------------------------------------------
        # RATING FEATURES
        # ---------------------------------------------

        self.rating_features = []

        # ---------------------------------------------
        # CLIENT RISK SETTINGS
        # ---------------------------------------------

        self.client_risk_enabled = True

        # Minimum rating influence from client risk.
        #
        # The client signal is blended with the ML
        # probability rather than completely replacing it.
        #
        # This makes client information independently
        # meaningful even when booking fields are unknown.
        self.client_risk_weight = 0.65

        # Additional hard protection against a 100%
        # bad-rate client being predicted as Excellent.
        self.client_rating_cap = True


    # =========================================================
    # TRAIN
    # =========================================================

    def train(
        self,
        train_df,
        val_df,
        test_df
    ):

        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()


        # =====================================================
        # ENSURE COMMENT
        # =====================================================

        for data in [
            train_df,
            val_df,
            test_df
        ]:

            if "comment" not in data.columns:

                data["comment"] = ""

            data["comment"] = (
                data["comment"]
                .fillna("")
                .astype(str)
            )


        # =====================================================
        # RATING FEATURES
        # =====================================================

        rating_features = [

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
            "client_bad_rate",
            "reservation_status",
            "is_canceled"

        ]

        self.rating_features = rating_features.copy()


        # =====================================================
        # STATUS FEATURES
        # =====================================================

        status_features = [

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
            "client_bad_rate",
            "is_canceled"

        ]

        self.status_features = status_features.copy()


        # =====================================================
        # ENSURE FEATURES EXIST
        # =====================================================

        for data in [
            train_df,
            val_df,
            test_df
        ]:

            for col in rating_features:

                if col not in data.columns:

                    if col in [
                        "is_top_bad_client",
                        "client_bad_rate",
                        "is_canceled"
                    ]:

                        data[col] = 0

                    else:

                        data[col] = "unknown"


            for col in status_features:

                if col not in data.columns:

                    if col in [
                        "is_top_bad_client",
                        "client_bad_rate",
                        "is_canceled"
                    ]:

                        data[col] = 0

                    else:

                        data[col] = "unknown"


        # =====================================================
        # NORMALISE NUMERIC FEATURES
        # =====================================================

        numeric_columns = [

            "is_top_bad_client",
            "client_bad_rate",
            "is_canceled"

        ]


        for data in [
            train_df,
            val_df,
            test_df
        ]:

            for col in numeric_columns:

                data[col] = pd.to_numeric(
                    data[col],
                    errors="coerce"
                )


        # =====================================================
        # CLIENT BAD RATE
        # =====================================================

        client_bad_rate_median = (
            train_df["client_bad_rate"]
            .median()
        )

        if pd.isna(client_bad_rate_median):

            client_bad_rate_median = 0.0


        for data in [
            train_df,
            val_df,
            test_df
        ]:

            data["client_bad_rate"] = (
                data["client_bad_rate"]
                .fillna(client_bad_rate_median)
                .clip(0, 1)
            )


        # =====================================================
        # NORMALISE IS CANCELED
        # =====================================================

        for data in [
            train_df,
            val_df,
            test_df
        ]:

            data["is_canceled"] = (
                data["is_canceled"]
                .fillna(0)
                .clip(0, 1)
            )


        # =====================================================
        # NORMALISE CLIENT SEGMENT
        # =====================================================

        for data in [
            train_df,
            val_df,
            test_df
        ]:

            data["client_segment"] = (
                data["client_segment"]
                .fillna("unknown")
                .astype(str)
                .str.strip()
                .str.lower()
            )


        # =====================================================
        # COMMENT TOPIC LEARNING
        # =====================================================

        print()
        print("==============================")
        print("COMMENT TOPIC LEARNING")
        print("==============================")


        self.comment_tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=3
        )


        self.comment_svd = TruncatedSVD(
            n_components=30,
            random_state=42
        )


        print(
            "Fitting comment TF-IDF on training data..."
        )


        train_comment_tfidf = (
            self.comment_tfidf.fit_transform(
                train_df["comment"]
            )
        )


        # -----------------------------------------------------
        # Protect SVD when training data is small.
        # -----------------------------------------------------

        max_components = min(
            30,
            max(
                1,
                train_comment_tfidf.shape[1] - 1
            )
        )

        self.comment_svd = TruncatedSVD(
            n_components=max_components,
            random_state=42
        )


        print(
            "Fitting comment SVD on training data..."
        )


        train_comment_features_array = (
            self.comment_svd.fit_transform(
                train_comment_tfidf
            )
        )


        val_comment_tfidf = (
            self.comment_tfidf.transform(
                val_df["comment"]
            )
        )


        val_comment_features_array = (
            self.comment_svd.transform(
                val_comment_tfidf
            )
        )


        test_comment_tfidf = (
            self.comment_tfidf.transform(
                test_df["comment"]
            )
        )


        test_comment_features_array = (
            self.comment_svd.transform(
                test_comment_tfidf
            )
        )


        comment_columns = [

            f"comment_topic_{i}"

            for i in range(
                train_comment_features_array.shape[1]
            )

        ]


        train_comment_features = pd.DataFrame(
            train_comment_features_array,
            columns=comment_columns,
            index=train_df.index
        )


        val_comment_features = pd.DataFrame(
            val_comment_features_array,
            columns=comment_columns,
            index=val_df.index
        )


        test_comment_features = pd.DataFrame(
            test_comment_features_array,
            columns=comment_columns,
            index=test_df.index
        )


        train_df = pd.concat(
            [
                train_df,
                train_comment_features
            ],
            axis=1
        )


        val_df = pd.concat(
            [
                val_df,
                val_comment_features
            ],
            axis=1
        )


        test_df = pd.concat(
            [
                test_df,
                test_comment_features
            ],
            axis=1
        )


        rating_features_with_comments = (
            rating_features +
            comment_columns
        )


        # =====================================================
        # FINAL RATING TARGET
        # =====================================================

        y_train = (
            pd.to_numeric(
                train_df["final_rating"],
                errors="coerce"
            )
            .fillna(0)
            .clip(0, 6)
            .astype(int)
        )


        y_val = (
            pd.to_numeric(
                val_df["final_rating"],
                errors="coerce"
            )
            .fillna(0)
            .clip(0, 6)
            .astype(int)
        )


        y_test = (
            pd.to_numeric(
                test_df["final_rating"],
                errors="coerce"
            )
            .fillna(0)
            .clip(0, 6)
            .astype(int)
        )


        # =====================================================
        # RESERVATION STATUS TARGET
        # =====================================================

        y_status_train = (
            train_df["reservation_status"]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
        )


        y_status_val = (
            val_df["reservation_status"]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
        )


        y_status_test = (
            test_df["reservation_status"]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
        )


        # =====================================================
        # RATING INPUT
        # =====================================================

        X_train = train_df[
            ["comment"] +
            rating_features_with_comments
        ]


        X_val = val_df[
            ["comment"] +
            rating_features_with_comments
        ]


        X_test = test_df[
            ["comment"] +
            rating_features_with_comments
        ]


        # =====================================================
        # CLASS WEIGHTS
        # =====================================================

        classes = np.unique(
            y_train
        )


        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train
        )


        class_weight_dict = dict(
            zip(
                classes,
                class_weights
            )
        )


        # =====================================================
        # TEXT PIPELINE
        # =====================================================

        text_pipeline = Pipeline([

            (
                "hash",
                HashingVectorizer(
                    n_features=2**16,
                    alternate_sign=False,
                    ngram_range=(1, 2)
                )
            ),

            (
                "tfidf",
                TfidfTransformer()
            )

        ])


        # =====================================================
        # RATING PREPROCESSOR
        # =====================================================

        rating_preprocess = ColumnTransformer([

            (
                "text",
                text_pipeline,
                "comment"
            ),

            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                [
                    c
                    for c in rating_features_with_comments
                    if c not in [
                        "is_top_bad_client",
                        "client_bad_rate",
                        "is_canceled"
                    ]
                ]
            ),

            (
                "num",
                "passthrough",
                [
                    c
                    for c in rating_features_with_comments
                    if c in [
                        "is_top_bad_client",
                        "client_bad_rate",
                        "is_canceled"
                    ]
                ]
            )

        ])


        # =====================================================
        # FINAL RATING MODELS
        # =====================================================

        models = {

            "Logistic Regression": Pipeline([

                (
                    "prep",
                    rating_preprocess
                ),

                (
                    "clf",
                    LogisticRegression(
                        solver="saga",
                        max_iter=500,
                        tol=1e-2,
                        class_weight=class_weight_dict,
                        n_jobs=-1
                    )
                )

            ]),


            "Linear SVC": Pipeline([

                (
                    "prep",
                    rating_preprocess
                ),

                (
                    "clf",
                    LinearSVC(
                        class_weight=class_weight_dict,
                        max_iter=3000,
                        tol=1e-2,
                        dual=False
                    )
                )

            ]),


            "LightGBM": Pipeline([

                (
                    "prep",
                    rating_preprocess
                ),

                (
                    "clf",
                    LGBMClassifier(
                        n_estimators=300,
                        learning_rate=0.05,
                        num_leaves=31,
                        class_weight=class_weight_dict,
                        random_state=42,
                        verbosity=-1
                    )
                )

            ])

        }


        self.models = models


        # =====================================================
        # TRAIN RATING MODELS
        # =====================================================

        print()
        print("==============================")
        print("FINAL RATING MODEL TRAINING")
        print("==============================")


        for name, model in models.items():

            print(
                f"Training {name}..."
            )


            model.fit(
                X_train,
                y_train
            )


            print(
                f"{name} completed."
            )


        # =====================================================
        # RESERVATION STATUS MODEL
        # =====================================================

        print()
        print("==============================")
        print("RESERVATION STATUS TRAINING")
        print("==============================")


        status_numeric_features = [

            "is_top_bad_client",
            "client_bad_rate",
            "is_canceled"

        ]


        status_categorical_features = [

            c
            for c in status_features
            if c not in status_numeric_features

        ]


        self.status_numeric_features = (
            status_numeric_features
        )


        self.status_categorical_features = (
            status_categorical_features
        )


        status_preprocess = ColumnTransformer([

            (
                "text",
                text_pipeline,
                "comment"
            ),

            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                status_categorical_features
            ),

            (
                "num",
                "passthrough",
                status_numeric_features
            )

        ])


        X_train_status = train_df[
            ["comment"] +
            status_features
        ]


        X_val_status = val_df[
            ["comment"] +
            status_features
        ]


        X_test_status = test_df[
            ["comment"] +
            status_features
        ]


        X_train_status_processed = (
            status_preprocess.fit_transform(
                X_train_status
            )
        )


        X_val_status_processed = (
            status_preprocess.transform(
                X_val_status
            )
        )


        X_test_status_processed = (
            status_preprocess.transform(
                X_test_status
            )
        )


        self.status_model = LogisticRegression(

            solver="saga",

            max_iter=500,

            tol=1e-2,

            class_weight="balanced",

            n_jobs=-1

        )


        print(
            "Training Reservation Status Model..."
        )


        self.status_model.fit(
            X_train_status_processed,
            y_status_train
        )


        self.status_preprocess = (
            status_preprocess
        )


        print(
            "Reservation Status Model completed."
        )


        # =====================================================
        # STATUS EVALUATION
        # =====================================================

        status_preds = (
            self.status_model.predict(
                X_test_status_processed
            )
        )


        print()
        print("==============================")
        print("RESERVATION STATUS EVALUATION")
        print("==============================")


        print(
            classification_report(
                y_status_test,
                status_preds,
                zero_division=0
            )
        )


        # =====================================================
        # VALIDATION
        # =====================================================

        print()
        print("==============================")
        print("FINAL RATING VALIDATION")
        print("==============================")


        validation_results = []


        for name, model in models.items():

            preds = model.predict(
                X_val
            )


            f1 = f1_score(
                y_val,
                preds,
                average="weighted"
            )


            try:

                clf = model.named_steps["clf"]


                if hasattr(
                    clf,
                    "predict_proba"
                ):

                    probs = (
                        model
                        .predict_proba(
                            X_val
                        )
                    )

                else:

                    scores = (
                        model
                        .decision_function(
                            X_val
                        )
                    )


                    scores = np.asarray(
                        scores
                    )


                    if scores.ndim == 1:

                        scores = np.column_stack(
                            [
                                -scores,
                                scores
                            ]
                        )


                    exp = np.exp(
                        scores -
                        np.max(
                            scores,
                            axis=1,
                            keepdims=True
                        )
                    )


                    probs = (
                        exp /
                        exp.sum(
                            axis=1,
                            keepdims=True
                        )
                    )


                roc = roc_auc_score(
                    y_val,
                    probs,
                    multi_class="ovr",
                    average="weighted"
                )


            except Exception as e:

                print(
                    f"[WARNING] ROC-AUC failed "
                    f"for {name}: {e}"
                )


                roc = 0.0


            validation_results.append({

                "model": name,

                "f1": f1,

                "roc": roc

            })


        validation_df = pd.DataFrame(
            validation_results
        )


        validation_df["final_score"] = (

            0.6 *
            validation_df["f1"]

            +

            0.4 *
            validation_df["roc"]

        )


        validation_df = (
            validation_df
            .sort_values(
                "final_score",
                ascending=False
            )
        )


        print(
            validation_df[
                [
                    "model",
                    "f1",
                    "roc",
                    "final_score"
                ]
            ].to_string(
                index=False
            )
        )


        # =====================================================
        # BEST RATING MODEL
        # =====================================================

        self.best_model_name = (
            validation_df.iloc[0]["model"]
        )


        self.best_model = (
            self.models[
                self.best_model_name
            ]
        )


        print(
            "\nBEST RATING MODEL SELECTED:",
            self.best_model_name
        )


        # =====================================================
        # REFERENCE DATA
        # =====================================================

        self.df_reference = pd.concat(
            [
                train_df,
                val_df,
                test_df
            ],
            axis=0
        )


        # =====================================================
        # COMMENT DATA
        # =====================================================

        reference_df = self.df_reference.copy()


        self.comment_df = reference_df[[

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
            "final_rating"

        ]].copy()


        # =====================================================
        # COMMENT INDEX
        # =====================================================

        self.comment_index = {}


        fields = [

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
            "guest_group"

        ]


        for rating, df_r in reference_df.groupby(
            "final_rating"
        ):

            rating_key = str(
                rating
            )


            self.comment_index[
                rating_key
            ] = {

                "ALL":
                    df_r[
                        "comment"
                    ].tolist(),

                "FIELDS":
                    {}

            }


            for field in fields:

                if field not in df_r.columns:
                    continue


                self.comment_index[
                    rating_key
                ][
                    "FIELDS"
                ][field] = (

                    df_r
                    .groupby(field)["comment"]
                    .apply(list)
                    .to_dict()

                )


        print(
            "FAST COMMENT INDEX READY"
        )


        # =====================================================
        # SAVE
        # =====================================================

        self._save_artifacts(
            reference_df
        )


        return reference_df


    # =========================================================
    # SAVE ARTIFACTS
    # =========================================================

    def _save_artifacts(
        self,
        df
    ):

        trained_dir = self.config.get(
            "trained_dir",
            "trained"
        )


        os.makedirs(
            trained_dir,
            exist_ok=True
        )


        train_path = self.config.get(
            "train_data",
            os.path.join(
                trained_dir,
                "train.parquet"
            )
        )


        df.to_parquet(
            train_path,
            index=False
        )


        print(
            f"[INFO] Training data saved: "
            f"{train_path}"
        )


        model_path = self.config.get(
            "service_model",
            os.path.join(
                trained_dir,
                "service_model.pkl"
            )
        )


        joblib.dump(
            self,
            model_path
        )


        print(
            f"[INFO] Service model saved: "
            f"{model_path}"
        )


        metadata_path = self.config.get(
            "training_metadata",
            os.path.join(
                trained_dir,
                "training_metadata.json"
            )
        )


        import json


        metadata = {}


        if os.path.exists(
            metadata_path
        ):

            with open(
                metadata_path,
                "r"
            ) as f:

                metadata = json.load(
                    f
                )


        metadata.update({

            "model":
                self.best_model_name,

            "trained_time":
                datetime.now().isoformat(),

            "rows":
                len(df),

            "columns":
                list(df.columns)

        })


        with open(
            metadata_path,
            "w"
        ) as f:

            json.dump(
                metadata,
                f,
                indent=4
            )


        print(
            f"[INFO] Metadata updated: "
            f"{metadata_path}"
        )


    # =========================================================
    # SWITCH MODEL
    # =========================================================

    def set_model(
        self,
        name: str
    ):

        if name in self.models:

            self.best_model_name = name

            self.best_model = (
                self.models[name]
            )


    # =========================================================
    # NORMALISE INPUT
    # =========================================================

    def _prepare_input(
        self,
        x
    ):

        input_df = pd.DataFrame(
            [x]
        )


        # ---------------------------------------------
        # Text
        # ---------------------------------------------

        if "comment" not in input_df.columns:

            input_df["comment"] = ""


        input_df["comment"] = (
            input_df["comment"]
            .fillna("")
            .astype(str)
        )


        # ---------------------------------------------
        # Categorical
        # ---------------------------------------------

        categorical_features = [

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
            "client_segment"

        ]


        for col in categorical_features:

            if col not in input_df.columns:

                input_df[col] = "unknown"


            input_df[col] = (
                input_df[col]
                .fillna("unknown")
                .astype(str)
                .str.strip()
                .str.lower()
            )


        # ---------------------------------------------
        # Numeric
        # ---------------------------------------------

        if "is_top_bad_client" not in input_df.columns:

            input_df["is_top_bad_client"] = 0


        if "is_canceled" not in input_df.columns:

            input_df["is_canceled"] = 0


        if "client_bad_rate" not in input_df.columns:

            input_df["client_bad_rate"] = np.nan


        input_df["is_top_bad_client"] = (
            pd.to_numeric(
                input_df["is_top_bad_client"],
                errors="coerce"
            )
            .fillna(0)
            .clip(0, 1)
        )


        input_df["is_canceled"] = (
            pd.to_numeric(
                input_df["is_canceled"],
                errors="coerce"
            )
            .fillna(0)
            .clip(0, 1)
        )


        # ---------------------------------------------
        # Client bad rate
        # ---------------------------------------------

        client_bad_rate = pd.to_numeric(
            input_df["client_bad_rate"],
            errors="coerce"
        )


        if client_bad_rate.isna().all():

            if (
                self.df_reference is not None
                and
                "client_bad_rate" in self.df_reference.columns
            ):

                fallback = pd.to_numeric(
                    self.df_reference[
                        "client_bad_rate"
                    ],
                    errors="coerce"
                ).median()


                if pd.isna(fallback):

                    fallback = 0.0

            else:

                fallback = 0.0


            client_bad_rate = (
                client_bad_rate
                .fillna(fallback)
            )

        else:

            fallback = client_bad_rate.median()


            if pd.isna(fallback):

                fallback = 0.0


            client_bad_rate = (
                client_bad_rate
                .fillna(fallback)
            )


        input_df["client_bad_rate"] = (
            client_bad_rate
            .clip(0, 1)
        )


        # ---------------------------------------------
        # Reservation status
        # ---------------------------------------------

        if "reservation_status" not in input_df.columns:

            input_df["reservation_status"] = "unknown"


        input_df["reservation_status"] = (
            input_df["reservation_status"]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
        )


        return input_df


    # =========================================================
    # CLIENT RISK SCORE
    # =========================================================

    def _calculate_client_risk(
        self,
        input_df
    ):

        bad_rate = float(
            input_df[
                "client_bad_rate"
            ].iloc[0]
        )


        top_bad = float(
            input_df[
                "is_top_bad_client"
            ].iloc[0]
        )


        segment = str(
            input_df[
                "client_segment"
            ].iloc[0]
        ).strip().lower()


        # ---------------------------------------------
        # Client bad rate
        # ---------------------------------------------

        bad_rate_score = (
            np.clip(
                bad_rate,
                0,
                1
            )
        )


        # ---------------------------------------------
        # Top bad client
        # ---------------------------------------------

        top_bad_score = (
            np.clip(
                top_bad,
                0,
                1
            )
        )


        # ---------------------------------------------
        # Client segment
        # ---------------------------------------------

        if segment in [
            "high-risk",
            "high risk",
            "bad",
            "poor",
            "very-high-risk",
            "very high risk"
        ]:

            segment_score = 1.0

        elif segment in [
            "medium-risk",
            "medium risk",
            "moderate-risk",
            "moderate risk"
        ]:

            segment_score = 0.5

        elif segment in [
            "low-risk",
            "low risk",
            "good",
            "normal"
        ]:

            segment_score = 0.0

        else:

            segment_score = 0.0


        # ---------------------------------------------
        # Combined client risk
        # ---------------------------------------------

        client_risk = (

            0.60 *
            bad_rate_score

            +

            0.25 *
            top_bad_score

            +

            0.15 *
            segment_score

        )


        return float(
            np.clip(
                client_risk,
                0,
                1
            )
        )


    # =========================================================
    # CLIENT RISK RATING PRIOR
    # =========================================================

    def _client_rating_prior(
        self,
        client_risk
    ):

        """
        Convert client risk into a rating probability.

        0.00 risk:
            mostly positive

        1.00 risk:
            mostly ratings 0-2

        This is intentionally independent from booking
        characteristics.
        """

        client_risk = float(
            np.clip(
                client_risk,
                0,
                1
            )
        )


        # Rating order: 0,1,2,3,4,5,6

        low_risk_prior = np.array([

            0.01,
            0.01,
            0.02,
            0.06,
            0.15,
            0.30,
            0.45

        ])


        high_risk_prior = np.array([

            0.25,
            0.25,
            0.22,
            0.15,
            0.08,
            0.04,
            0.01

        ])


        prior = (

            (
                1.0 -
                client_risk
            )
            *
            low_risk_prior

            +

            client_risk
            *
            high_risk_prior

        )


        prior = (
            prior /
            prior.sum()
        )


        return prior


    # =========================================================
    # RATING PROBABILITIES
    # =========================================================

    def _get_rating_probabilities(
        self,
        X_rating
    ):

        clf = (
            self.best_model
            .named_steps["clf"]
        )


        if hasattr(
            clf,
            "predict_proba"
        ):

            probabilities = (
                self.best_model
                .predict_proba(
                    X_rating
                )[0]
            )

        else:

            scores = (
                self.best_model
                .decision_function(
                    X_rating
                )
            )


            scores = np.asarray(
                scores
            )


            if scores.ndim == 2:

                scores = scores[0]


            exp = np.exp(
                scores -
                np.max(scores)
            )


            probabilities = (
                exp /
                exp.sum()
            )


        classes = list(
            clf.classes_
        )


        rating_probabilities = np.zeros(
            7,
            dtype=float
        )


        for probability, cls in zip(
            probabilities,
            classes
        ):

            try:

                rating_class = int(
                    float(cls)
                )

            except Exception:

                continue


            if 0 <= rating_class <= 6:

                rating_probabilities[
                    rating_class
                ] = float(
                    probability
                )


        total = rating_probabilities.sum()


        if total <= 0:

            rating_probabilities = (
                np.ones(7) / 7
            )

        else:

            rating_probabilities = (
                rating_probabilities /
                total
            )


        return rating_probabilities


    # =========================================================
    # APPLY CLIENT RISK TO RATING
    # =========================================================

    def _apply_client_risk_to_rating(
        self,
        ml_probabilities,
        client_risk
    ):

        client_prior = (
            self._client_rating_prior(
                client_risk
            )
        )


        weight = float(
            np.clip(
                self.client_risk_weight,
                0,
                1
            )
        )


        final_probabilities = (

            (
                1.0 -
                weight
            )
            *
            ml_probabilities

            +

            weight
            *
            client_prior

        )


        final_probabilities = (
            final_probabilities /
            final_probabilities.sum()
        )


        # =====================================================
        # HARD CLIENT-RISK PROTECTION
        # =====================================================
        #
        # This prevents the exact problem in the logs:
        #
        # client_bad_rate = 1.0
        # is_top_bad_client = 1
        # client_segment = high-risk
        #
        # -> rating 6 / Excellent
        #
        # A client with 100% historical bad rate should not
        # receive an Excellent prediction simply because the
        # other booking fields are unknown.
        # =====================================================

        if self.client_rating_cap:

            if client_risk >= 0.90:

                # Rating 5 and 6 cannot dominate.
                final_probabilities[5] *= 0.10
                final_probabilities[6] *= 0.02


            elif client_risk >= 0.70:

                final_probabilities[6] *= 0.05


            elif client_risk >= 0.50:

                final_probabilities[6] *= 0.20


            final_probabilities = (
                final_probabilities /
                final_probabilities.sum()
            )


        return final_probabilities


    # =========================================================
    # PREDICT
    # =========================================================

    def predict(
        self,
        x: dict
    ):

        # =====================================================
        # PREPARE UI INPUT
        # =====================================================

        input_df = self._prepare_input(
            x
        )


        # =====================================================
        # CLIENT RISK
        # =====================================================

        client_risk = (
            self._calculate_client_risk(
                input_df
            )
        )


        print()
        print("==============================")
        print("CLIENT RISK SIGNAL")
        print("==============================")


        print(
            "Client Segment:",
            input_df[
                "client_segment"
            ].iloc[0]
        )


        print(
            "Client Bad Rate:",
            input_df[
                "client_bad_rate"
            ].iloc[0]
        )


        print(
            "Top Bad Client:",
            input_df[
                "is_top_bad_client"
            ].iloc[0]
        )


        print(
            "Calculated Client Risk:",
            round(
                client_risk,
                4
            )
        )


        # =====================================================
        # COMMENT TOPIC FEATURES
        # =====================================================

        if (
            self.comment_tfidf is None
            or self.comment_svd is None
        ):

            raise RuntimeError(
                "Comment TF-IDF/SVD models are not available. "
                "Please retrain the ServiceModel."
            )


        comment_tfidf = (
            self.comment_tfidf.transform(
                input_df["comment"]
            )
        )


        comment_features_array = (
            self.comment_svd.transform(
                comment_tfidf
            )
        )


        comment_columns = [

            f"comment_topic_{i}"

            for i in range(
                comment_features_array.shape[1]
            )

        ]


        comment_features = pd.DataFrame(

            comment_features_array,

            columns=comment_columns,

            index=input_df.index

        )


        input_df = pd.concat(
            [
                input_df,
                comment_features
            ],
            axis=1
        )


        # =====================================================
        # FINAL RATING
        # =====================================================

        rating_features_with_comments = (

            self.rating_features +

            comment_columns

        )


        X_rating = input_df[
            ["comment"] +
            rating_features_with_comments
        ]


        # =====================================================
        # RAW ML RATING PROBABILITIES
        # =====================================================

        ml_rating_probabilities = (
            self._get_rating_probabilities(
                X_rating
            )
        )


        # =====================================================
        # CLIENT-ADJUSTED RATING PROBABILITIES
        # =====================================================

        rating_probabilities = (
            self._apply_client_risk_to_rating(
                ml_rating_probabilities,
                client_risk
            )
        )


        # =====================================================
        # FINAL PREDICTED RATING
        # =====================================================

        pred_rating = int(
            np.argmax(
                rating_probabilities
            )
        )


        pred_rating = int(
            np.clip(
                pred_rating,
                0,
                6
            )
        )


        # =====================================================
        # RATING RISK
        # =====================================================

        rating_risk_map = {

            0: 100,
            1: 90,
            2: 75,
            3: 60,
            4: 40,
            5: 20,
            6: 10

        }


        risk_weights = {

            0: 1.00,
            1: 0.90,
            2: 0.75,
            3: 0.60,
            4: 0.40,
            5: 0.20,
            6: 0.00

        }


        sla_risk = 0.0


        for rating_class in range(7):

            sla_risk += (

                rating_probabilities[
                    rating_class
                ]

                *

                risk_weights[
                    rating_class
                ]

            )


        sla_risk_score = round(
            sla_risk * 100,
            1
        )


        base_risk = rating_risk_map.get(
            pred_rating,
            50
        )


        risk_score = round(

            (
                0.6 *
                base_risk
            )

            +

            (
                0.4 *
                sla_risk_score
            ),

            1

        )


        # =====================================================
        # RESERVATION STATUS
        # =====================================================

        status_input = input_df[

            [
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
                "client_bad_rate",
                "is_canceled"

            ]

        ]


        X_status = (
            self.status_preprocess.transform(
                status_input
            )
        )


        status_probs = (
            self.status_model
            .predict_proba(
                X_status
            )[0]
        )


        status_classes = (
            self.status_model.classes_
        )


        # =====================================================
        # DEBUG STATUS MODEL
        # =====================================================

        print()
        print(
            "STATUS MODEL CLASSES:",
            list(status_classes)
        )


        print(
            "STATUS MODEL RAW PROBABILITIES:",
            status_probs
        )


        # =====================================================
        # INITIALISE
        # =====================================================

        prob_canceled = 0.0
        prob_check_out = 0.0
        prob_no_show = 0.0


        # =====================================================
        # STATUS CLASS NORMALISATION
        # =====================================================

        for probability, cls in zip(
            status_probs,
            status_classes
        ):

            normalized_cls = (
                str(cls)
                .strip()
                .lower()
                .replace("_", "-")
                .replace(" ", "-")
            )


            percentage = (
                float(probability) *
                100.0
            )


            # -------------------------------------------------
            # Text classes
            # -------------------------------------------------

            if normalized_cls in [
                "canceled",
                "cancelled"
            ]:

                prob_canceled = percentage


            elif normalized_cls in [
                "check-out",
                "checkout",
                "check_out"
            ]:

                prob_check_out = percentage


            elif normalized_cls in [
                "no-show",
                "noshow",
                "no_show"
            ]:

                prob_no_show = percentage


            # -------------------------------------------------
            # Numeric encoded classes
            #
            # Expected mapping:
            #
            # 0 = Canceled
            # 1 = Check-Out
            # 2 = No-Show
            # -------------------------------------------------

            else:

                try:

                    numeric_cls = float(
                        cls
                    )


                    if numeric_cls == 0:

                        prob_canceled = percentage


                    elif numeric_cls == 1:

                        prob_check_out = percentage


                    elif numeric_cls == 2:

                        prob_no_show = percentage


                except Exception:

                    pass


        # =====================================================
        # STATUS DEBUG
        # =====================================================

        print()
        print(
            "STATUS MAPPED PROBABILITIES:"
        )


        print(
            "Cancellation:",
            prob_canceled
        )


        print(
            "Check-Out:",
            prob_check_out
        )


        print(
            "No-Show:",
            prob_no_show
        )


        # =====================================================
        # IS CANCELED SIGNAL
        # =====================================================

        is_canceled_value = float(
            input_df[
                "is_canceled"
            ].iloc[0]
        )


        if is_canceled_value >= 1:

            if prob_canceled < 1.0:

                prob_canceled = 1.0


        # =====================================================
        # NORMALISE STATUS PROBABILITIES
        # =====================================================

        status_total = (

            prob_canceled +

            prob_check_out +

            prob_no_show

        )


        if status_total > 0:

            prob_canceled = (

                prob_canceled /
                status_total *
                100

            )


            prob_check_out = (

                prob_check_out /
                status_total *
                100

            )


            prob_no_show = (

                prob_no_show /
                status_total *
                100

            )


        # =====================================================
        # ROUND
        # =====================================================

        prob_canceled = round(
            prob_canceled,
            1
        )


        prob_check_out = round(
            prob_check_out,
            1
        )


        prob_no_show = round(
            prob_no_show,
            1
        )


        # =====================================================
        # COMMENT
        # =====================================================

        generated_comment = (
            self._generate_comment(
                pred_rating,
                sla_risk_score,
                risk_score,
                x,
                client_risk=client_risk
            )
        )


        # =====================================================
        # DEBUG INFORMATION
        # =====================================================

        print()
        print("==============================")
        print("FINAL PREDICTION")
        print("==============================")


        print(
            "ML Predicted Rating:",
            int(
                np.argmax(
                    ml_rating_probabilities
                )
            )
        )


        print(
            "Client Adjusted Rating:",
            pred_rating
        )


        print(
            "Client Risk:",
            round(
                client_risk,
                4
            )
        )


        print(
            "Rating Risk:",
            risk_score
        )


        print(
            "SLA Risk:",
            sla_risk_score
        )


        print(
            "Client Bad Rate:",
            input_df[
                "client_bad_rate"
            ].iloc[0]
        )


        print()
        print(
            "Reservation Status Probabilities:"
        )


        print(
            "Cancellation:",
            prob_canceled,
            "%"
        )


        print(
            "Check-Out:",
            prob_check_out,
            "%"
        )


        print(
            "No-Show:",
            prob_no_show,
            "%"
        )


        # =====================================================
        # RETURN
        # =====================================================

        return (

            pred_rating,

            prob_canceled,

            prob_check_out,

            prob_no_show,

            sla_risk_score,

            risk_score,

            generated_comment

        )


    # =========================================================
    # COMMENT GENERATION
    # =========================================================

    def _generate_comment(
        self,
        rating,
        sla_risk,
        risk,
        x,
        client_risk=0.0
    ):

        msg = []


        # =====================================================
        # CLIENT RISK OVERRIDE
        # =====================================================
        #
        # The comment must reflect client risk.
        #
        # This prevents:
        #
        # 100% bad client
        # ->
        # "Excellent service."
        #
        # even if the raw ML model originally selected 6.
        # =====================================================

        if client_risk >= 0.90:

            msg.append(
                "High-risk customer with a strong "
                "history of poor outcomes."
            )


        elif client_risk >= 0.70:

            msg.append(
                "Customer shows a high level of "
                "historical service risk."
            )


        elif client_risk >= 0.50:

            msg.append(
                "Customer shows moderate service risk."
            )


        else:

            # =================================================
            # RATING 0-6
            # =================================================

            if rating == 6:

                msg.append(
                    "Excellent service."
                )

            elif rating == 5:

                msg.append(
                    "Very good service."
                )

            elif rating == 4:

                msg.append(
                    "Positive service."
                )

            elif rating == 3:

                msg.append(
                    "Average service."
                )

            elif rating == 2:

                msg.append(
                    "Service quality issue."
                )

            elif rating == 1:

                msg.append(
                    "Poor service experience."
                )

            else:

                msg.append(
                    "Severe dissatisfaction."
                )


        # =====================================================
        # HISTORICAL COMMENTS
        # =====================================================

        context_comments = (
            self._get_top_comments_dynamic(
                rating,
                x,
                top_k=20
            )
        )


        if context_comments:

            ai_summary = (
                self._generate_ai_summary(
                    context_comments,
                    rating
                )
            )


            if ai_summary:

                msg.append(
                    ai_summary
                )


            example_block = [
                "Examples of similar feedback:"
            ]


            example_block.extend(

                [
                    f"{i}. {c}"

                    for i, c in enumerate(
                        context_comments[:5],
                        1
                    )

                ]

            )


            msg.append(
                "\n".join(
                    example_block
                )
            )


        else:

            msg.append(
                "No historical patterns found; "
                "using client risk and operational signals."
            )


        return "\n\n".join(
            msg
        )


    # =========================================================
    # TOP COMMENTS
    # =========================================================

    def _get_top_comments_dynamic(
        self,
        rating,
        x,
        top_k=5
    ):

        if self.comment_df is None:

            return []


        df = self.comment_df.copy()


        df["final_rating"] = (
            pd.to_numeric(
                df["final_rating"],
                errors="coerce"
            )
        )


        df = df[
            df["final_rating"] == rating
        ]


        if len(df) == 0:

            return []


        priority_fields = [

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
            "guest_group"

        ]


        filtered = df


        for field in priority_fields:

            if field not in filtered.columns:

                continue


            value = x.get(
                field
            )


            if value in [
                None,
                "",
                "unknown"
            ]:

                continue


            tmp = filtered[
                filtered[field].astype(str)
                ==
                str(value)
            ]


            if len(tmp) > 0:

                filtered = tmp


        comments = (
            filtered["comment"]
            .dropna()
            .astype(str)
        )


        comments = comments[
            comments.apply(
                is_valid_comment
            )
        ]


        comments = (
            comments
            .value_counts()
            .index
            .tolist()
        )


        if len(comments) == 0:

            comments = (
                df["comment"]
                .dropna()
                .astype(str)
            )


            comments = comments[
                comments.apply(
                    is_valid_comment
                )
            ]


            comments = (
                comments
                .value_counts()
                .index
                .tolist()
            )


        return comments[:top_k]


    # =========================================================
    # AI SUMMARY
    # =========================================================

    def _generate_ai_summary(
        self,
        comments,
        rating
    ):

        if not comments:

            return None


        try:

            corpus = [

                str(x).lower()

                for x in comments[:30]

            ]


            vectorizer = CountVectorizer(

                stop_words="english",

                ngram_range=(2, 3),

                max_features=50,

                min_df=2

            )


            X = vectorizer.fit_transform(
                corpus
            )


            phrases = (
                vectorizer
                .get_feature_names_out()
            )


            freq = np.asarray(
                X.sum(
                    axis=0
                )
            ).ravel()


            df_kw = pd.DataFrame({

                "phrase": phrases,

                "freq": freq

            })


            def is_valid_phrase(p):

                words = p.split()


                if len(set(words)) == 1:

                    return False


                if len(words) == 1:

                    return False


                if any(

                    w in [
                        "service",
                        "customer"
                    ]

                    for w in words

                ):

                    return False


                return True


            df_kw = df_kw[
                df_kw["phrase"].apply(
                    is_valid_phrase
                )
            ]


            df_kw = df_kw.sort_values(
                "freq",
                ascending=False
            )


            top_phrases = (
                df_kw[
                    "phrase"
                ]
                .head(5)
                .tolist()
            )


            if not top_phrases:

                return None


            keyword_text = (
                "; ".join(
                    top_phrases
                )
            )


            # =================================================
            # LOW RATINGS
            # =================================================

            if rating == 0:

                return (
                    "Customers frequently reported "
                    "severe problems involving "
                    f"{keyword_text}."
                )


            elif rating == 1:

                return (
                    "Customers frequently reported "
                    "poor experiences involving "
                    f"{keyword_text}."
                )


            elif rating == 2:

                return (
                    "Customers frequently highlighted "
                    "service issues involving "
                    f"{keyword_text}."
                )


            elif rating == 3:

                return (
                    "Mixed experiences were observed, "
                    "with recurring themes such as "
                    f"{keyword_text}."
                )


            elif rating == 4:

                return (
                    "Feedback was generally positive, "
                    "with mentions of "
                    f"{keyword_text}."
                )


            elif rating == 5:

                return (
                    "Customers frequently highlighted "
                    "very good experiences such as "
                    f"{keyword_text}."
                )


            else:

                return (
                    "Customers frequently highlighted "
                    "excellent experiences such as "
                    f"{keyword_text}."
                )


        except Exception:

            return None




