# =========================================================
# src/model.py
# =========================================================

import os
import json
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

        # =====================================================
        # FINAL RATING MODEL
        # =====================================================

        self.best_model = None
        self.best_model_name = None

        self.models = {}
        self.results_df = None

        # =====================================================
        # REFERENCE DATA
        # =====================================================

        self.df_reference = None

        # =====================================================
        # CUSTOMER EMAIL HISTORY
        # =====================================================

        self.customer_email_index = {}

        # =====================================================
        # COMMENT MODEL
        # =====================================================

        self.comment_df = None
        self.comment_index = {}

        self.comment_tfidf = None
        self.comment_svd = None

        # =====================================================
        # RESERVATION STATUS MODEL
        # =====================================================

        self.status_model = None
        self.status_preprocess = None

        self.status_features = []
        self.status_numeric_features = []
        self.status_categorical_features = []

        # =====================================================
        # RESERVATION STATUS MAPPING
        #
        # 0 -> canceled
        # 1 -> no-show
        # 2 -> check-out
        # =====================================================

        self.status_class_mapping = {

            0: "canceled",
            1: "no-show",
            2: "check-out"

        }

        # =====================================================
        # RATING FEATURES
        # =====================================================

        self.rating_features = []

        # =====================================================
        # RATING NUMERIC FEATURES
        # =====================================================

        self.rating_numeric_features = []

        self.rating_categorical_features = []


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
        # ENSURE CLIENT EMAIL
        # =====================================================

        for data in [
            train_df,
            val_df,
            test_df
        ]:

            if "client_email" not in data.columns:
                data["client_email"] = ""

            data["client_email"] = (
                data["client_email"]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
            )

        # =====================================================
        # RATING FEATURES
        #
        # HIGH PRIORITY:
        #
        # is_top_bad_client
        # client_bad_rate
        # bad_client_score
        #
        # Revenue features remain training features.
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

            # HIGH PRIORITY CUSTOMER RISK
            "is_top_bad_client",
            "client_bad_rate",
            "bad_client_score",

            "reservation_status",
            "is_canceled",

            # TRAINING FEATURES
            "revenue_cancellation_exposure",
            "revenue_at_risk",
            "total_nights_group"

        ]

        self.rating_features = rating_features.copy()

        # =====================================================
        # STATUS FEATURES
        #
        # reservation_status is NOT included.
        # is_canceled is NOT included.
        #
        # Revenue features remain training features.
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

            # CUSTOMER RISK
            "is_top_bad_client",
            "client_bad_rate",

            # TRAINING FEATURES
            "revenue_cancellation_exposure",
            "revenue_at_risk",
            "total_nights_group"

        ]

        self.status_features = status_features.copy()

        # =====================================================
        # ENSURE FEATURES EXIST
        # =====================================================

        numeric_defaults = [

            "is_top_bad_client",
            "client_bad_rate",
            "bad_client_score",
            "is_canceled",
            "revenue_cancellation_exposure",
            "revenue_at_risk"

        ]

        for data in [
            train_df,
            val_df,
            test_df
        ]:

            for col in rating_features:

                if col not in data.columns:

                    if col in numeric_defaults:

                        data[col] = 0.0

                    else:

                        data[col] = "unknown"

            for col in status_features:

                if col not in data.columns:

                    if col in numeric_defaults:

                        data[col] = 0.0

                    else:

                        data[col] = "unknown"

        # =====================================================
        # NORMALISE NUMERIC FEATURES
        # =====================================================

        numeric_columns = [

            "is_top_bad_client",
            "client_bad_rate",
            "bad_client_score",
            "is_canceled",
            "revenue_cancellation_exposure",
            "revenue_at_risk"

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
        # IS TOP BAD CLIENT
        # =====================================================

        for data in [
            train_df,
            val_df,
            test_df
        ]:

            data["is_top_bad_client"] = (
                data["is_top_bad_client"]
                .fillna(0)
                .clip(0, 1)
            )

        # =====================================================
        # BAD CLIENT SCORE
        #
        # is_top_bad_client has 70% weight.
        # client_bad_rate has 30% weight.
        #
        # This makes the explicit bad-client flag stronger.
        # =====================================================

        for data in [
            train_df,
            val_df,
            test_df
        ]:

            data["bad_client_score"] = (

                0.70 *
                data["is_top_bad_client"]

                +

                0.30 *
                data["client_bad_rate"]

            ).clip(0, 1)

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
        # NORMALISE REVENUE FEATURES
        # =====================================================

        for data in [
            train_df,
            val_df,
            test_df
        ]:

            data["revenue_cancellation_exposure"] = (
                data["revenue_cancellation_exposure"]
                .fillna(0.0)
                .clip(lower=0)
            )

            data["revenue_at_risk"] = (
                data["revenue_at_risk"]
                .fillna(0.0)
                .clip(lower=0)
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
            .astype(int)
        )

        y_val = (
            pd.to_numeric(
                val_df["final_rating"],
                errors="coerce"
            )
            .fillna(0)
            .astype(int)
        )

        y_test = (
            pd.to_numeric(
                test_df["final_rating"],
                errors="coerce"
            )
            .fillna(0)
            .astype(int)
        )

        # =====================================================
        # RESERVATION STATUS TARGET
        #
        # 0 = canceled
        # 1 = no-show
        # 2 = check-out
        # =====================================================

        def normalise_status_target(series):

            numeric = pd.to_numeric(
                series,
                errors="coerce"
            )

            return numeric.astype("Int64")

        y_status_train = normalise_status_target(
            train_df["reservation_status"]
        )

        y_status_val = normalise_status_target(
            val_df["reservation_status"]
        )

        y_status_test = normalise_status_target(
            test_df["reservation_status"]
        )

        train_status_mask = (
            y_status_train.isin([0, 1, 2])
        )

        val_status_mask = (
            y_status_val.isin([0, 1, 2])
        )

        test_status_mask = (
            y_status_test.isin([0, 1, 2])
        )

        y_status_train = (
            y_status_train[
                train_status_mask
            ]
            .astype(int)
        )

        y_status_val = (
            y_status_val[
                val_status_mask
            ]
            .astype(int)
        )

        y_status_test = (
            y_status_test[
                test_status_mask
            ]
            .astype(int)
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

        self.rating_numeric_features = [

            "is_top_bad_client",
            "client_bad_rate",
            "bad_client_score",
            "is_canceled",
            "revenue_cancellation_exposure",
            "revenue_at_risk"

        ]

        self.rating_categorical_features = [

            c

            for c in rating_features_with_comments

            if c not in self.rating_numeric_features

        ]

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
                self.rating_categorical_features
            ),

            (
                "num",
                "passthrough",
                self.rating_numeric_features
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
                        max_iter=300,
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
                        max_iter=2000,
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
            "revenue_cancellation_exposure",
            "revenue_at_risk"

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

        X_train_status_all = train_df[
            ["comment"] +
            status_features
        ]

        X_val_status_all = val_df[
            ["comment"] +
            status_features
        ]

        X_test_status_all = test_df[
            ["comment"] +
            status_features
        ]

        X_train_status = (
            X_train_status_all.loc[
                train_status_mask
            ]
        )

        X_val_status = (
            X_val_status_all.loc[
                val_status_mask
            ]
        )

        X_test_status = (
            X_test_status_all.loc[
                test_status_mask
            ]
        )

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
            max_iter=300,
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

                    probs = model.predict_proba(
                        X_val
                    )

                else:

                    scores = model.decision_function(
                        X_val
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
        ).copy()

        # =====================================================
        # BUILD CUSTOMER EMAIL INDEX
        # =====================================================

        self._build_customer_email_index()

        # =====================================================
        # COMMENT DATA
        # =====================================================

        reference_df = (
            self.df_reference.copy()
        )

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
    # BUILD CUSTOMER EMAIL INDEX
    # =========================================================

    def _build_customer_email_index(self):

        self.customer_email_index = {}

        if self.df_reference is None:
            return

        if "client_email" not in self.df_reference.columns:
            return

        reference = self.df_reference.copy()

        reference["client_email"] = (
            reference["client_email"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        for email, group in reference.groupby(
            "client_email"
        ):

            email = str(email).strip().lower()

            if not email:
                continue

            history_count = len(group)

            bad_rate = 0.0

            if "is_top_bad_client" in group.columns:

                bad_flags = pd.to_numeric(
                    group["is_top_bad_client"],
                    errors="coerce"
                ).fillna(0)

                bad_rate = float(
                    bad_flags.mean()
                )

            historical_client_bad_rate = np.nan

            if "client_bad_rate" in group.columns:

                rates = pd.to_numeric(
                    group["client_bad_rate"],
                    errors="coerce"
                ).dropna()

                if len(rates) > 0:

                    historical_client_bad_rate = float(
                        rates.mean()
                    )

            avg_rating = np.nan

            if "final_rating" in group.columns:

                ratings = pd.to_numeric(
                    group["final_rating"],
                    errors="coerce"
                ).dropna()

                if len(ratings) > 0:

                    avg_rating = float(
                        ratings.mean()
                    )

            cancel_rate = np.nan

            if "is_canceled" in group.columns:

                cancellations = pd.to_numeric(
                    group["is_canceled"],
                    errors="coerce"
                ).fillna(0)

                cancel_rate = float(
                    cancellations.mean()
                )

            self.customer_email_index[email] = {

                "history_count":
                    history_count,

                "bad_client_rate":
                    bad_rate,

                "historical_client_bad_rate":
                    historical_client_bad_rate,

                "avg_rating":
                    avg_rating,

                "cancel_rate":
                    cancel_rate

            }

        print(
            f"[INFO] Customer email index ready: "
            f"{len(self.customer_email_index)} customers"
        )


    # =========================================================
    # CUSTOMER EMAIL LOOKUP
    # =========================================================

    def _lookup_customer_email(
        self,
        email
    ):

        if email is None:
            return None

        email = (
            str(email)
            .strip()
            .lower()
        )

        if not email:
            return None

        return self.customer_email_index.get(
            email
        )


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

        # =====================================================
        # CLIENT EMAIL
        #
        # Email is an identity field.
        # It is NOT directly one-hot encoded.
        # =====================================================

        if "client_email" not in input_df.columns:

            input_df["client_email"] = ""

        input_df["client_email"] = (
            input_df["client_email"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        # =====================================================
        # COMMENT
        # =====================================================

        if "comment" not in input_df.columns:

            input_df["comment"] = ""

        input_df["comment"] = (
            input_df["comment"]
            .fillna("")
            .astype(str)
        )

        # =====================================================
        # CATEGORICAL
        # =====================================================

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
            "client_segment",
            "reservation_status"

        ]

        for col in categorical_features:

            if col not in input_df.columns:

                input_df[col] = "unknown"

            input_df[col] = (
                input_df[col]
                .fillna("unknown")
                .astype(str)
                .str.strip()
            )

        # =====================================================
        # NUMERIC FEATURES
        # =====================================================

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

        # =====================================================
        # OPTIONAL REVENUE / STAY FEATURES
        #
        # These are required by the trained model structure,
        # but NOT required from predict_page.py.
        # =====================================================

        if "revenue_cancellation_exposure" not in input_df.columns:

            input_df[
                "revenue_cancellation_exposure"
            ] = 0.0

        if "revenue_at_risk" not in input_df.columns:

            input_df[
                "revenue_at_risk"
            ] = 0.0

        if "total_nights_group" not in input_df.columns:

            input_df[
                "total_nights_group"
            ] = "unknown"

        input_df[
            "revenue_cancellation_exposure"
        ] = pd.to_numeric(
            input_df[
                "revenue_cancellation_exposure"
            ],
            errors="coerce"
        ).fillna(0.0)

        input_df[
            "revenue_at_risk"
        ] = pd.to_numeric(
            input_df[
                "revenue_at_risk"
            ],
            errors="coerce"
        ).fillna(0.0)

        input_df[
            "total_nights_group"
        ] = (
            input_df[
                "total_nights_group"
            ]
            .fillna("unknown")
            .astype(str)
        )

        # =====================================================
        # CLIENT BAD RATE
        # =====================================================

        client_bad_rate = pd.to_numeric(

            input_df["client_bad_rate"],

            errors="coerce"

        )

        if client_bad_rate.isna().all():

            fallback = 0.0

            if (
                self.df_reference is not None
                and
                "client_bad_rate"
                in self.df_reference.columns
            ):

                fallback = pd.to_numeric(

                    self.df_reference[
                        "client_bad_rate"
                    ],

                    errors="coerce"

                ).median()

                if pd.isna(
                    fallback
                ):

                    fallback = 0.0

            client_bad_rate = (
                client_bad_rate.fillna(
                    fallback
                )
            )

        else:

            fallback = (
                client_bad_rate.median()
            )

            if pd.isna(
                fallback
            ):

                fallback = 0.0

            client_bad_rate = (
                client_bad_rate.fillna(
                    fallback
                )
            )

        input_df["client_bad_rate"] = (
            client_bad_rate.clip(0, 1)
        )

        # =====================================================
        # EMAIL CUSTOMER HISTORY
        #
        # Email has priority as the customer identity.
        #
        # We use the historical email record to strengthen
        # the bad-client signal.
        # =====================================================

        email = (
            input_df[
                "client_email"
            ].iloc[0]
        )

        email_history = (
            self._lookup_customer_email(
                email
            )
        )

        input_df["email_known"] = 0.0

        input_df["email_history_count"] = 0.0

        input_df["email_history_bad_rate"] = 0.0

        input_df["email_history_avg_rating"] = np.nan

        input_df["email_history_cancel_rate"] = np.nan

        if email_history is not None:

            input_df["email_known"] = 1.0

            input_df["email_history_count"] = float(
                email_history.get(
                    "history_count",
                    0
                )
            )

            input_df["email_history_bad_rate"] = float(
                email_history.get(
                    "bad_client_rate",
                    0.0
                )
            )

            input_df["email_history_avg_rating"] = (
                email_history.get(
                    "avg_rating",
                    np.nan
                )
            )

            input_df["email_history_cancel_rate"] = (
                email_history.get(
                    "cancel_rate",
                    np.nan
                )
            )

        # =====================================================
        # EMAIL-BASED BAD CLIENT PRIORITY
        #
        # If the email belongs to a historically bad client,
        # strengthen the explicit bad-client signals.
        #
        # IMPORTANT:
        # The email does not directly become a rating.
        # It only provides customer-history evidence.
        # =====================================================

        if email_history is not None:

            email_bad_rate = float(
                email_history.get(
                    "bad_client_rate",
                    0.0
                )
            )

            historical_bad_rate = email_history.get(
                "historical_client_bad_rate",
                np.nan
            )

            if not pd.isna(
                historical_bad_rate
            ):

                email_bad_rate = max(
                    email_bad_rate,
                    float(historical_bad_rate)
                )

            if email_bad_rate >= 0.80:

                input_df[
                    "client_bad_rate"
                ] = np.maximum(
                    input_df[
                        "client_bad_rate"
                    ],
                    email_bad_rate
                )

            if email_bad_rate >= 1.0:

                input_df[
                    "is_top_bad_client"
                ] = np.maximum(
                    input_df[
                        "is_top_bad_client"
                    ],
                    1.0
                )

        # =====================================================
        # BAD CLIENT SCORE
        # =====================================================

        input_df["bad_client_score"] = (

            0.70 *
            input_df["is_top_bad_client"]

            +

            0.30 *
            input_df["client_bad_rate"]

        ).clip(0, 1)

        # =====================================================
        # NORMALISE RESERVATION STATUS
        # =====================================================

        reservation_status = str(

            input_df[
                "reservation_status"
            ].iloc[0]

        ).strip().lower()

        if reservation_status in [
            "cancelled",
            "canceled"
        ]:

            reservation_status = "canceled"

        elif reservation_status in [
            "checkout",
            "check out",
            "check-out"
        ]:

            reservation_status = "check-out"

        elif reservation_status in [
            "noshow",
            "no show",
            "no-show"
        ]:

            reservation_status = "no-show"

        elif reservation_status in [
            "",
            "nan",
            "none",
            "null",
            "unknown"
        ]:

            reservation_status = "unknown"

        input_df[
            "reservation_status"
        ] = reservation_status

        # =====================================================
        # KEEP EXPLICIT CANCELLATION SIGNAL CONSISTENT
        # =====================================================

        if reservation_status == "canceled":

            input_df[
                "is_canceled"
            ] = 1.0

        elif reservation_status in [
            "no-show",
            "check-out"
        ]:

            input_df[
                "is_canceled"
            ] = 0.0

        return input_df


    # =========================================================
    # APPLY CUSTOMER PRIORITY TO RATING
    # =========================================================

    def _apply_customer_rating_priority(
        self,
        pred_rating,
        input_df
    ):

        pred_rating = int(
            pred_rating
        )

        email = str(
            input_df[
                "client_email"
            ].iloc[0]
        ).strip().lower()

        is_top_bad_client = float(
            input_df[
                "is_top_bad_client"
            ].iloc[0]
        )

        client_bad_rate = float(
            input_df[
                "client_bad_rate"
            ].iloc[0]
        )

        bad_client_score = float(
            input_df[
                "bad_client_score"
            ].iloc[0]
        )

        email_known = float(
            input_df[
                "email_known"
            ].iloc[0]
        )

        email_bad_rate = float(
            input_df[
                "email_history_bad_rate"
            ].iloc[0]
        )

        # =====================================================
        # DEBUG
        # =====================================================

        print()
        print("==============================")
        print("CUSTOMER PRIORITY")
        print("==============================")

        print(
            "Client Email:",
            email
        )

        print(
            "Email Known:",
            email_known
        )

        print(
            "Email Historical Bad Rate:",
            email_bad_rate
        )

        print(
            "is_top_bad_client:",
            is_top_bad_client
        )

        print(
            "client_bad_rate:",
            client_bad_rate
        )

        print(
            "bad_client_score:",
            bad_client_score
        )

        print(
            "ML Rating Before Customer Priority:",
            pred_rating
        )

        # =====================================================
        # PRIORITY 1
        #
        # EMAIL + HISTORICAL BAD CLIENT
        #
        # A known email with a very high historical bad rate
        # receives the strongest customer-risk treatment.
        # =====================================================

        if (
            email_known >= 1.0
            and
            email_bad_rate >= 0.80
        ):

            pred_rating = min(
                pred_rating,
                3
            )

            print(
                "[CUSTOMER RULE] Known high-risk email "
                "-> maximum rating 3"
            )

        # =====================================================
        # PRIORITY 2
        #
        # EXPLICIT TOP BAD CLIENT
        # =====================================================

        elif is_top_bad_client >= 1.0:

            pred_rating = min(
                pred_rating,
                3
            )

            print(
                "[CUSTOMER RULE] is_top_bad_client=1 "
                "-> maximum rating 3"
            )

        # =====================================================
        # PRIORITY 3
        #
        # HIGH CLIENT BAD RATE
        # =====================================================

        elif client_bad_rate >= 0.80:

            pred_rating = min(
                pred_rating,
                4
            )

            print(
                "[CUSTOMER RULE] client_bad_rate>=0.80 "
                "-> maximum rating 4"
            )

        # =====================================================
        # PRIORITY 4
        #
        # MODERATE BAD RATE
        # =====================================================

        elif client_bad_rate >= 0.60:

            pred_rating = min(
                pred_rating,
                5
            )

            print(
                "[CUSTOMER RULE] client_bad_rate>=0.60 "
                "-> maximum rating 5"
            )

        pred_rating = int(
            np.clip(
                pred_rating,
                0,
                6
            )
        )

        print(
            "Final Rating After Customer Priority:",
            pred_rating
        )

        return pred_rating


    # =========================================================
    # PREDICT
    # =========================================================

    def predict(
        self,
        x: dict
    ):

        # =====================================================
        # PREPARE INPUT
        # =====================================================

        input_df = self._prepare_input(
            x
        )

        # =====================================================
        # COMMENT TOPIC FEATURES
        # =====================================================

        if (
            self.comment_tfidf is None
            or
            self.comment_svd is None
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
        # PART 1
        # FINAL RATING PREDICTION
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
        # ML RATING
        # =====================================================

        ml_pred_rating = int(

            self.best_model.predict(
                X_rating
            )[0]

        )

        ml_pred_rating = int(

            np.clip(
                ml_pred_rating,
                0,
                6
            )

        )

        print()
        print("==============================")
        print("ML RATING")
        print("==============================")

        print(
            "ML Predicted Rating:",
            ml_pred_rating
        )

        # =====================================================
        # CUSTOMER PRIORITY
        # =====================================================

        pred_rating = (
            self._apply_customer_rating_priority(
                ml_pred_rating,
                input_df
            )
        )

        # =====================================================
        # RATING CLASSIFIER
        # =====================================================

        clf = (
            self.best_model
            .named_steps["clf"]
        )

        # =====================================================
        # RATING PROBABILITIES
        # =====================================================

        if hasattr(
            clf,
            "predict_proba"
        ):

            rating_probs = (

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

            rating_probs = (

                exp /
                np.sum(exp)

            )

        rating_class_order = list(
            clf.classes_
        )

        # =====================================================
        # SLA RISK
        # =====================================================

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

        for i, cls in enumerate(
            rating_class_order
        ):

            if i >= len(
                rating_probs
            ):

                continue

            try:

                rating_class = int(
                    cls
                )

            except Exception:

                continue

            sla_risk += (

                rating_probs[i]

                *

                risk_weights.get(
                    rating_class,
                    0.50
                )

            )

        sla_risk_score = round(

            sla_risk * 100,

            1

        )

        # =====================================================
        # CUSTOMER RISK SCORE
        # =====================================================

        rating_risk_map = {

            0: 100,
            1: 90,
            2: 75,
            3: 60,
            4: 40,
            5: 20,
            6: 0

        }

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
        # PART 2
        # RESERVATION STATUS PREDICTION
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
                "revenue_cancellation_exposure",
                "revenue_at_risk",
                "total_nights_group"
            ]

        ]

        # =====================================================
        # TRANSFORM STATUS INPUT
        # =====================================================

        X_status = (

            self.status_preprocess.transform(
                status_input
            )

        )

        # =====================================================
        # STATUS PROBABILITIES
        # =====================================================

        status_probs = (

            self.status_model
            .predict_proba(
                X_status
            )[0]

        )

        status_classes = (

            self.status_model.classes_

        )

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
        # MAP STATUS PROBABILITIES
        # =====================================================

        ml_prob_canceled = 0.0
        ml_prob_no_show = 0.0
        ml_prob_check_out = 0.0

        for probability, cls in zip(

            status_probs,
            status_classes

        ):

            try:

                status_class = int(
                    float(cls)
                )

            except Exception:

                continue

            probability = (

                float(probability)

                *

                100.0

            )

            if status_class == 0:

                ml_prob_canceled = (
                    probability
                )

            elif status_class == 1:

                ml_prob_no_show = (
                    probability
                )

            elif status_class == 2:

                ml_prob_check_out = (
                    probability
                )

        # =====================================================
        # READ EXPLICIT INPUTS
        # =====================================================

        reservation_status_input = str(

            input_df[
                "reservation_status"
            ].iloc[0]

        ).strip().lower()

        is_canceled_value = float(

            input_df[
                "is_canceled"
            ].iloc[0]

        )

        # =====================================================
        # STATUS DEBUG
        # =====================================================

        print()
        print(
            "STATUS ML MAPPING:"
        )

        print(
            "ML Cancellation:",
            ml_prob_canceled
        )

        print(
            "ML No-Show:",
            ml_prob_no_show
        )

        print(
            "ML Check-Out:",
            ml_prob_check_out
        )

        # =====================================================
        # FINAL STATUS DECISION
        #
        # 1. is_canceled
        # 2. explicit reservation_status
        # 3. ML
        # =====================================================

        if is_canceled_value >= 1.0:

            prob_canceled = 100.0
            prob_no_show = 0.0
            prob_check_out = 0.0

        elif reservation_status_input == "canceled":

            prob_canceled = 100.0
            prob_no_show = 0.0
            prob_check_out = 0.0

        elif reservation_status_input == "no-show":

            prob_canceled = 0.0
            prob_no_show = 100.0
            prob_check_out = 0.0

        elif reservation_status_input == "check-out":

            prob_canceled = 0.0
            prob_no_show = 0.0
            prob_check_out = 100.0

        else:

            prob_canceled = ml_prob_canceled
            prob_no_show = ml_prob_no_show
            prob_check_out = ml_prob_check_out

        # =====================================================
        # STATUS NORMALISATION
        # =====================================================

        status_total = (

            prob_canceled +

            prob_no_show +

            prob_check_out

        )

        if status_total > 0:

            prob_canceled = (

                prob_canceled /
                status_total *
                100.0

            )

            prob_no_show = (

                prob_no_show /
                status_total *
                100.0

            )

            prob_check_out = (

                prob_check_out /
                status_total *
                100.0

            )

        else:

            prob_canceled = 0.0
            prob_no_show = 0.0
            prob_check_out = 0.0

        # =====================================================
        # ROUND
        # =====================================================

        prob_canceled = round(
            prob_canceled,
            1
        )

        prob_no_show = round(
            prob_no_show,
            1
        )

        prob_check_out = round(
            prob_check_out,
            1
        )

        # =====================================================
        # FINAL STATUS DEBUG
        # =====================================================

        print()
        print(
            "RESERVATION STATUS INPUT:",
            reservation_status_input
        )

        print(
            "IS CANCELED INPUT:",
            is_canceled_value
        )

        print()
        print(
            "STATUS MAPPED PROBABILITIES:"
        )

        print(
            "Cancellation:",
            prob_canceled
        )

        print(
            "No-Show:",
            prob_no_show
        )

        print(
            "Check-Out:",
            prob_check_out
        )

        # =====================================================
        # COMMENT
        # =====================================================

        generated_comment = (

            self._generate_comment(

                pred_rating,

                sla_risk_score,

                risk_score,

                x

            )

        )

        # =====================================================
        # FINAL DEBUG
        # =====================================================

        print()
        print("==============================")
        print("FINAL PREDICTION")
        print("==============================")

        print(
            "ML Predicted Rating:",
            ml_pred_rating
        )

        print(
            "Final Predicted Rating:",
            pred_rating
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
            "Client Email:",
            input_df[
                "client_email"
            ].iloc[0]
        )

        print(
            "is_top_bad_client:",
            input_df[
                "is_top_bad_client"
            ].iloc[0]
        )

        print(
            "Client Bad Rate:",
            input_df[
                "client_bad_rate"
            ].iloc[0]
        )

        print(
            "Bad Client Score:",
            input_df[
                "bad_client_score"
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
            "No-Show:",
            prob_no_show,
            "%"
        )

        print(
            "Check-Out:",
            prob_check_out,
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
        x

    ):

        msg = []

        # =====================================================
        # RATING
        # =====================================================

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
                "using operational signals."

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

                filtered[field]
                .astype(str)

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

            df_kw = (

                df_kw
                .sort_values(
                    "freq",
                    ascending=False
                )

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

            if rating == 6:

                return (

                    "Customers frequently "
                    "highlighted excellent "
                    f"experiences such as "
                    f"{keyword_text}."

                )

            elif rating == 5:

                return (

                    "Customers frequently "
                    "highlighted very good "
                    f"experiences such as "
                    f"{keyword_text}."

                )

            elif rating == 4:

                return (

                    "Feedback was generally "
                    "positive, with mentions "
                    f"of {keyword_text}."

                )

            elif rating == 3:

                return (

                    "Mixed experiences were "
                    "observed, with recurring "
                    f"themes such as "
                    f"{keyword_text}."

                )

            elif rating == 2:

                return (

                    "Customers reported service "
                    "issues involving "
                    f"{keyword_text}."

                )

            elif rating == 1:

                return (

                    "Poor experiences were "
                    "associated with issues "
                    f"such as {keyword_text}."

                )

            else:

                return (

                    "Severe dissatisfaction was "
                    "associated with issues "
                    f"such as {keyword_text}."

                )

        except Exception:

            return None





