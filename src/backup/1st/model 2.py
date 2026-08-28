# =========================================================
# src/model.py
# =========================================================

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
# COMMENT VALIDATION HELPERS
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


class ServiceModel:

    def __init__(self, config=None):

        self.config = config or {}

        self.best_model = None
        self.best_model_name = None

        self.models = {}
        self.results_df = None

        self.df_reference = None

        self.comment_df = None
        self.comment_index = {}

        # ---------------------------------
        # Reservation status model
        # ---------------------------------

        self.status_model = None
        self.status_preprocess = None

        # ---------------------------------
        # Comment topic model
        # ---------------------------------

        self.comment_tfidf = None
        self.comment_svd = None


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
        # ENSURE COMMENT COLUMN
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
        # STRUCTURED FEATURES
        # =====================================================

        structured_features = [

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


        # =====================================================
        # MAKE SURE STRUCTURED FEATURES EXIST
        # =====================================================

        for data in [
            train_df,
            val_df,
            test_df
        ]:

            for col in structured_features:

                if col not in data.columns:

                    data[col] = "unknown"


        # =====================================================
        # COMMENT TOPIC LEARNING
        # =====================================================

        print()
        print("==============================")
        print("COMMENT TOPIC LEARNING")
        print("==============================")


        # -----------------------------------------------------
        # Increased from 30 to 50 topics
        # -----------------------------------------------------

        self.comment_tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=3,
            sublinear_tf=True
        )


        self.comment_svd = TruncatedSVD(
            n_components=50,
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


        print(
            "Transforming validation comments..."
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


        print(
            "Transforming test comments..."
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


        # =====================================================
        # ATTACH COMMENT FEATURES
        # =====================================================

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


        structured_features_with_comments = (
            structured_features +
            comment_columns
        )


        # =====================================================
        # TARGETS
        # =====================================================

        y_train = train_df[
            "final_rating"
        ]

        y_val = val_df[
            "final_rating"
        ]

        y_test = test_df[
            "final_rating"
        ]


        y_status_train = train_df[
            "reservation_status"
        ]

        y_status_val = val_df[
            "reservation_status"
        ]

        y_status_test = test_df[
            "reservation_status"
        ]


        # =====================================================
        # FEATURES
        # =====================================================

        X_train = train_df[
            ["comment"] +
            structured_features_with_comments
        ]

        X_val = val_df[
            ["comment"] +
            structured_features_with_comments
        ]

        X_test = test_df[
            ["comment"] +
            structured_features_with_comments
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


        print()
        print("==============================")
        print("DATA SPLIT")
        print("==============================")


        print(
            "Training rows   :",
            len(X_train)
        )

        print(
            "Validation rows:",
            len(X_val)
        )

        print(
            "Test rows       :",
            len(X_test)
        )


        print()
        print("Rating distribution:")

        print(
            y_train.value_counts()
            .sort_index()
            .to_string()
        )


        # =====================================================
        # TEXT PIPELINE
        #
        # IMPORTANT CHANGE:
        #
        # HashingVectorizer has been replaced with
        # TfidfVectorizer.
        # =====================================================

        text_pipeline = TfidfVectorizer(

            max_features=15000,

            ngram_range=(1, 2),

            min_df=2,

            max_df=0.95,

            sublinear_tf=True,

            strip_accents="unicode"

        )


        # =====================================================
        # PREPROCESSOR
        # =====================================================

        preprocess = ColumnTransformer(

            [

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
                    structured_features_with_comments
                )

            ],

            remainder="drop"

        )


        # =====================================================
        # MODELS
        # =====================================================

        models = {


            # =================================================
            # LOGISTIC REGRESSION
            # =================================================

            "Logistic Regression": Pipeline([

                (
                    "prep",
                    preprocess
                ),

                (
                    "clf",
                    LogisticRegression(

                        solver="saga",

                        C=1.0,

                        max_iter=1000,

                        tol=1e-4,

                        class_weight=class_weight_dict,

                        n_jobs=-1,

                        random_state=42

                    )
                )

            ]),


            # =================================================
            # LINEAR SVC
            # =================================================

            "Linear SVC": Pipeline([

                (
                    "prep",
                    preprocess
                ),

                (
                    "clf",
                    LinearSVC(

                        C=1.0,

                        class_weight=class_weight_dict,

                        max_iter=5000,

                        tol=1e-4,

                        dual=False,

                        random_state=42

                    )
                )

            ]),


            # =================================================
            # LIGHTGBM
            # =================================================

            "LightGBM": Pipeline([

                (
                    "prep",
                    preprocess
                ),

                (
                    "clf",
                    LGBMClassifier(

                        n_estimators=500,

                        learning_rate=0.03,

                        num_leaves=31,

                        max_depth=-1,

                        min_child_samples=20,

                        subsample=0.8,

                        colsample_bytree=0.8,

                        class_weight=class_weight_dict,

                        random_state=42,

                        verbosity=-1

                    )
                )

            ])

        }


        self.models = models


        # =====================================================
        # EVALUATION HELPER
        # =====================================================

        def evaluate(
            model,
            X_eval,
            y_true,
            name,
            show_report=True
        ):

            # -------------------------------------------------
            # CLASS PREDICTIONS
            # -------------------------------------------------

            preds = model.predict(
                X_eval
            )


            # -------------------------------------------------
            # F1
            # -------------------------------------------------

            f1 = f1_score(

                y_true,

                preds,

                average="weighted"

            )


            # -------------------------------------------------
            # ROC-AUC
            #
            # IMPORTANT:
            #
            # For probability models:
            #     use predict_proba()
            #
            # For LinearSVC:
            #     use RAW decision_function()
            #
            # DO NOT apply softmax to SVC scores.
            # -------------------------------------------------

            try:

                clf = model.named_steps[
                    "clf"
                ]


                if hasattr(
                    clf,
                    "predict_proba"
                ):

                    roc_scores = (
                        model.predict_proba(
                            X_eval
                        )
                    )


                elif hasattr(
                    clf,
                    "decision_function"
                ):

                    roc_scores = (
                        model.decision_function(
                            X_eval
                        )
                    )

                    roc_scores = np.asarray(
                        roc_scores
                    )


                else:

                    raise ValueError(
                        "Model does not provide "
                        "predict_proba or "
                        "decision_function"
                    )


                # -------------------------------------------------
                # Multiclass ROC-AUC
                # -------------------------------------------------

                roc = roc_auc_score(

                    y_true,

                    roc_scores,

                    multi_class="ovr",

                    average="weighted",

                    labels=clf.classes_

                )


            except Exception as e:

                print(

                    f"[WARNING] ROC-AUC failed "
                    f"for {name}: {e}"

                )

                roc = 0.0


            # -------------------------------------------------
            # REPORT
            # -------------------------------------------------

            if show_report:

                print()
                print("==============================")
                print(name)
                print("==============================")


                print(
                    classification_report(
                        y_true,
                        preds
                    )
                )


                print(
                    "F1 Score :",
                    round(
                        f1,
                        4
                    )
                )


                print(
                    "ROC-AUC  :",
                    round(
                        roc,
                        4
                    )
                )


            return {

                "model": name,

                "f1": f1,

                "roc": roc

            }


        # =====================================================
        # TRAIN MODELS
        # =====================================================

        print()
        print("==============================")
        print("TRAINING MODELS")
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


        status_text_pipeline = TfidfVectorizer(

            max_features=15000,

            ngram_range=(1, 2),

            min_df=2,

            max_df=0.95,

            sublinear_tf=True,

            strip_accents="unicode"

        )


        status_preprocess = ColumnTransformer([

            (
                "text",
                status_text_pipeline,
                "comment"
            ),

            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                structured_features_with_comments
            )

        ])


        print(
            "Fitting reservation status preprocessing..."
        )


        X_train_status = (
            status_preprocess.fit_transform(
                X_train
            )
        )


        X_val_status = (
            status_preprocess.transform(
                X_val
            )
        )


        X_test_status = (
            status_preprocess.transform(
                X_test
            )
        )


        self.status_model = LogisticRegression(

            solver="saga",

            C=1.0,

            max_iter=1000,

            tol=1e-4,

            class_weight="balanced",

            n_jobs=-1,

            random_state=42

        )


        print(
            "Training Reservation Status Model..."
        )


        self.status_model.fit(
            X_train_status,
            y_status_train
        )


        print(
            "Reservation Status Model completed."
        )


        self.status_preprocess = (
            status_preprocess
        )


        # =====================================================
        # RESERVATION STATUS EVALUATION
        # =====================================================

        print()
        print("==============================")
        print("RESERVATION STATUS EVALUATION")
        print("==============================")


        status_preds = (
            self.status_model.predict(
                X_test_status
            )
        )


        print(
            classification_report(
                y_status_test,
                status_preds
            )
        )


        # =====================================================
        # VALIDATION
        # =====================================================

        print()
        print("==============================")
        print("VALIDATION")
        print("==============================")


        validation_results = []


        for name, model in models.items():

            result = evaluate(

                model,

                X_val,

                y_val,

                name,

                show_report=True

            )


            validation_results.append(
                result
            )


        # =====================================================
        # SELECT BEST MODEL
        #
        # ROC HAS NOW BEEN GIVEN MORE WEIGHT.
        #
        # Old:
        # 0.6 F1 + 0.4 ROC
        #
        # New:
        # 0.4 F1 + 0.6 ROC
        # =====================================================

        validation_df = pd.DataFrame(
            validation_results
        )


        validation_df["final_score"] = (

            0.4 *
            validation_df["f1"]

            +

            0.6 *
            validation_df["roc"]

        )


        validation_df = validation_df.sort_values(

            "final_score",

            ascending=False

        )


        print()
        print("==============================")
        print("VALIDATION MODEL RANKING")
        print("==============================")


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
        # BEST MODEL
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
            "\nBEST MODEL SELECTED:",
            self.best_model_name
        )


        # =====================================================
        # FINAL TEST EVALUATION
        # =====================================================

        print()
        print("==============================")
        print("FINAL TEST EVALUATION")
        print("==============================")


        test_result = evaluate(

            self.best_model,

            X_test,

            y_test,

            self.best_model_name,

            show_report=True

        )


        self.results_df = pd.DataFrame([

            {

                "model":
                    self.best_model_name,

                "f1":
                    test_result["f1"],

                "roc":
                    test_result["roc"],

                "final_score":
                    (
                        0.4 *
                        test_result["f1"]

                        +

                        0.6 *
                        test_result["roc"]
                    )

            }

        ])


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


        reference_df["comment"] = (

            reference_df["comment"]

            .fillna("")

            .astype(str)

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
            "FAST COMMENT INDEX READY "
            "(RATING-FIRST + UI FILTER)"
        )


        # =====================================================
        # SAVE ARTIFACTS
        # =====================================================

        self._save_artifacts(
            reference_df
        )


        return reference_df


    # =========================================================
    # SAVE MODEL + TRAIN DATA
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


        # =====================================================
        # SAVE PROCESSED TRAIN DATA
        # =====================================================

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


        # =====================================================
        # SAVE SERVICE MODEL
        # =====================================================

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


        # =====================================================
        # SAVE METADATA
        # =====================================================

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
    # PREDICT
    # =========================================================

    def predict(
        self,
        x: dict
    ):

        # =====================================================
        # STRUCTURED FEATURES
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


        # =====================================================
        # CREATE INPUT DATAFRAME
        # =====================================================

        input_df = pd.DataFrame(
            [x]
        )


        # =====================================================
        # ENSURE STRUCTURED FEATURES EXIST
        # =====================================================

        for col in structured_features:

            if col not in input_df.columns:

                input_df[col] = "unknown"


        # =====================================================
        # COMMENT
        # =====================================================

        input_df["comment"] = (

            input_df["comment"]

            .fillna("")

            .astype(str)

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

                "Comment TF-IDF/SVD models "
                "are not available. "
                "Please retrain the ServiceModel."

            )


        # =====================================================
        # TRANSFORM COMMENT
        # =====================================================

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


        # =====================================================
        # CREATE COMMENT TOPIC COLUMNS
        # =====================================================

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


        # =====================================================
        # ATTACH COMMENT TOPICS
        # =====================================================

        input_df = pd.concat(

            [
                input_df,
                comment_features
            ],

            axis=1

        )


        # =====================================================
        # FINAL FEATURE LIST
        # =====================================================

        structured_features_with_comments = [

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

        ] + comment_columns


        # =====================================================
        # FEATURES
        # =====================================================

        X_input = input_df[

            ["comment"] +

            structured_features_with_comments

        ]


        # =====================================================
        # RATING PREDICTION
        # =====================================================

        pred_rating = int(

            self.best_model.predict(

                X_input

            )[0]

        )


        clf = (

            self.best_model

            .named_steps["clf"]

        )


        # =====================================================
        # RESERVATION STATUS
        # =====================================================

        X_status = (

            self.status_preprocess.transform(

                X_input

            )

        )


        status_probs = (

            self.status_model.predict_proba(

                X_status

            )[0]

        )


        status_class_order = (

            self.status_model.classes_

        )


        prob_canceled = 0.0
        prob_check_out = 0.0
        prob_no_show = 0.0


        for i, cls in enumerate(

            status_class_order

        ):

            probability = (

                status_probs[i] *

                100

            )


            if cls == "canceled":

                prob_canceled = round(

                    probability,

                    1

                )


            elif cls == "check-out":

                prob_check_out = round(

                    probability,

                    1

                )


            elif cls == "no-show":

                prob_no_show = round(

                    probability,

                    1

                )


        # =====================================================
        # RATING PROBABILITY
        #
        # IMPORTANT:
        #
        # LinearSVC uses raw decision scores.
        # We still convert them to a normalized distribution
        # here because your SLA-risk calculation requires
        # values that sum to 1.
        #
        # This does NOT affect ROC-AUC evaluation.
        # =====================================================

        if hasattr(

            clf,

            "predict_proba"

        ):

            probs = (

                self.best_model

                .predict_proba(

                    X_input

                )[0]

            )


        else:

            scores = (

                self.best_model

                .decision_function(

                    X_input

                )

            )


            scores = np.asarray(
                scores
            )


            if scores.ndim == 2:

                scores = scores[0]


            # ---------------------------------------------
            # Numerically stable softmax
            #
            # ONLY used for SLA risk prediction.
            # NOT used for ROC-AUC.
            # ---------------------------------------------

            scores = (

                scores -

                np.max(scores)

            )


            exp_scores = np.exp(
                scores
            )


            probs = (

                exp_scores /

                np.sum(
                    exp_scores
                )

            )


        # =====================================================
        # CLASS ORDER
        # =====================================================

        class_order = list(

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

            class_order

        ):

            if i >= len(probs):

                continue


            sla_risk += (

                probs[i] *

                risk_weights.get(

                    cls,

                    0

                )

            )


        sla_risk_score = round(

            sla_risk *

            100,

            1

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


        base_risk = rating_map.get(

            pred_rating,

            50

        )


        risk_score = round(

            0.6 *

            base_risk

            +

            0.4 *

            sla_risk_score,

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

                x

            )

        )


        # =====================================================
        # RETURN
        #
        # KEEPING YOUR EXISTING 7 VALUES
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


        if rating == 5:

            msg.append(
                "Excellent service."
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

        else:

            msg.append(
                "High dissatisfaction."
            )


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

                filtered[field] == value

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


            if rating == 5:

                return (

                    "Customers frequently "

                    "highlighted positive "

                    "experiences such as "

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

                    "themes such as "

                    f"{keyword_text}."

                )


            elif rating == 2:

                return (

                    "Customers reported service "

                    "issues involving "

                    f"{keyword_text}."

                )


            else:

                return (

                    "Severe dissatisfaction was "

                    "associated with issues "

                    f"such as {keyword_text}."

                )


        except Exception:

            return None



