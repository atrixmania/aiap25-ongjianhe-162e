# =========================================================
# src/preprocess.py
# =========================================================

import numpy as np
import pandas as pd
import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

import warnings
from sklearn.exceptions import ConvergenceWarning


warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning
)


# =========================================================
# DATA PROCESSOR
# =========================================================

class DataProcessor:

    # =====================================================
    # AUTHORITATIVE CATEGORICAL MAPPINGS
    # =====================================================
    #
    # SINGLE SOURCE OF TRUTH
    #
    # Do NOT generate these mappings from the dataframe.
    # Do NOT sort them alphabetically.
    # Do NOT create another mapping elsewhere.
    #
    # These exact values are used for:
    #
    #     1. Encoding during preprocessing
    #     2. Reverse mapping during EDA
    #     3. Categorical range validation
    #
    # =====================================================

    AUTHORITATIVE_MAPPINGS = {

        "hotel": {
            "airport hotel": 0,
            "city hotel": 1
        },

        "meal": {
            "BB (Bed & Breakfast)": 0,
            "FB (Full Board)": 1,
            "HB (Half Board)": 2,
            "SC (Self Catering)": 3,
            "missing": 4,
            "undefined": 5
        },

        "market_segment": {
            "aviation": 0,
            "complementary": 1,
            "corporate": 2,
            "direct": 3,
            "groups": 4,
            "missing": 5,
            "offline-ta/to": 6,
            "online-ta": 7,
            "undefined": 8
        },

        "distribution_channel": {
            "corporate": 0,
            "direct": 1,
            "gds": 2,
            "missing": 3,
            "ta/to": 4,
            "undefined": 5
        },

        "deposit_type": {
            "no-deposit": 0,
            "non-refund": 1,
            "refundable": 2
        },

        "customer_type": {
            "contract": 0,
            "group": 1,
            "missing": 2,
            "transient": 3,
            "transient-party": 4
        },

        "reservation_status": {
            "canceled": 0,
            "check-out": 1,
            "no-show": 2
        }
    }


    # =====================================================
    # AUTOMATIC REVERSE MAPPINGS
    # =====================================================
    #
    # Example:
    #
    #     "airport hotel": 0
    #
    # becomes:
    #
    #     0: "airport hotel"
    #
    # This is generated ONLY from
    # AUTHORITATIVE_MAPPINGS.
    #
    # =====================================================

    REVERSE_MAPPINGS = {

        col: {
            encoded_value: original_value
            for original_value, encoded_value
            in mapping.items()
        }

        for col, mapping
        in AUTHORITATIVE_MAPPINGS.items()
    }


    # =====================================================
    # INITIALIZATION
    # =====================================================

    def __init__(self, config=None):

        self.config = config or {}

        # -------------------------------------------------
        # Normal categorical encoders
        # -------------------------------------------------

        self.encoders = {}


        # -------------------------------------------------
        # Rating model
        # -------------------------------------------------

        self.rating_model = None

        self.rating_classes = []


        # -------------------------------------------------
        # Iterative imputers
        # -------------------------------------------------

        self.numeric_imputer = None

        self.categorical_imputer = None

        self.numeric_imputer_columns = []

        self.categorical_imputer_columns = []


        # -------------------------------------------------
        # Encoded categorical columns
        # -------------------------------------------------

        self.categorical_cols = [

            "hotel",
            "meal",
            "market_segment",
            "distribution_channel",
            "deposit_type",
            "customer_type",
            "reservation_status"

        ]


        # -------------------------------------------------
        # Raw categorical / text columns
        # -------------------------------------------------

        self.raw_categorical_cols = [

            "country",
            "agent",
            "company",
            "promo_code",
            "comment"

        ]


        # -------------------------------------------------
        # ID / text columns
        # -------------------------------------------------

        self.id_text_cols = [

            "client_email"

        ]


        # -------------------------------------------------
        # Valid encoded categorical ranges
        # -------------------------------------------------

        self.categorical_ranges = {

            col: (
                min(mapping.values()),
                max(mapping.values())
            )

            for col, mapping
            in self.AUTHORITATIVE_MAPPINGS.items()

        }


    # =====================================================
    # PREPARE FINAL RATING
    # =====================================================

    def prepare_final_rating(self, df):

        df = df.copy()

        df = self._clean_data(df)

        if "rating" not in df.columns:

            raise KeyError(
                "Column 'rating' is required "
                "to create final_rating."
            )

        df["rating"] = pd.to_numeric(
            df["rating"],
            errors="coerce"
        )

        invalid_rating = (

            df["rating"].notna()

            &

            ~df["rating"].isin(
                [0, 1, 2, 3, 4, 5, 6]
            )

        )

        if invalid_rating.any():

            print(
                "[WARNING] Invalid rating values "
                "outside 0-6 found."
            )

            print(
                df.loc[
                    invalid_rating,
                    "rating"
                ].value_counts()
            )

            df.loc[
                invalid_rating,
                "rating"
            ] = np.nan


        # -------------------------------------------------
        # Prepare comments
        # -------------------------------------------------

        if "comment" not in df.columns:

            df["comment"] = ""

        else:

            df["comment"] = (
                df["comment"]
                .fillna("")
                .astype(str)
            )


        # -------------------------------------------------
        # Training data
        # -------------------------------------------------

        train = df[
            df["rating"].notna()
        ].copy()

        if len(train) == 0:

            raise ValueError(
                "No valid ratings are available "
                "to train the rating model."
            )


        X = train["comment"]

        y = train["rating"].astype(int)


        self.rating_classes = sorted(
            y.unique().tolist()
        )

        print(
            "[INFO] Rating classes:"
        )

        print(
            self.rating_classes
        )


        # -------------------------------------------------
        # Rating model
        # -------------------------------------------------

        self.rating_model = Pipeline([

            (
                "tfidf",

                TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2)
                )
            ),

            (
                "clf",

                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced"
                )
            )

        ])


        self.rating_model.fit(
            X,
            y
        )


        print(
            "[INFO] Rating model fitted."
        )


        # -------------------------------------------------
        # Predict missing ratings
        # -------------------------------------------------

        df["rating_predicted"] = np.nan

        missing_mask = (
            df["rating"].isna()
        )

        if missing_mask.sum() > 0:

            predicted = (
                self.rating_model.predict(
                    df.loc[
                        missing_mask,
                        "comment"
                    ]
                )
            )

            df.loc[
                missing_mask,
                "rating_predicted"
            ] = predicted


        # -------------------------------------------------
        # Final rating
        # -------------------------------------------------

        df["final_rating"] = (

            df["rating"]

            .fillna(
                df["rating_predicted"]
            )

        )


        df["final_rating"] = pd.to_numeric(
            df["final_rating"],
            errors="coerce"
        )


        df.loc[
            ~df["final_rating"].isin(
                [0, 1, 2, 3, 4, 5, 6]
            ),
            "final_rating"
        ] = np.nan


        df["final_rating"] = (
            df["final_rating"]
            .round()
            .astype("Int64")
        )


        print(
            "[INFO] Rating target preparation completed."
        )

        print(
            f"[INFO] Original rating missing: "
            f"{df['rating'].isna().sum()}"
        )

        print(
            "[INFO] final_rating distribution:"
        )

        print(
            df["final_rating"]
            .value_counts()
            .sort_index()
        )

        print(
            "[INFO] final_rating created."
        )


        return df


    # =====================================================
    # FIT NORMAL PREPROCESSOR
    # =====================================================

    def fit(self, train_df):

        df = train_df.copy()

        df = self._clean_data(df)

        self._fit_encoders(df)

        df = self._encode(df)

        self._fit_imputer(df)

        print(
            "DataProcessor fitted"
        )

        return self


    # =====================================================
    # TRANSFORM
    # =====================================================

    def transform(self, df):

        df = df.copy()

        df = self._clean_data(df)

        df = self._encode(df)

        if "final_rating" not in df.columns:

            df = self._predict_missing_rating(df)

        df = self._impute(df)

        print(
            "DataProcessor transform completed"
        )

        return df


    # =====================================================
    # FIT TRANSFORM
    # =====================================================

    def fit_transform(self, train_df):

        self.fit(train_df)

        return self.transform(
            train_df
        )


    # =====================================================
    # STEP 4.1 CLEAN DATA
    # =====================================================

    def _clean_data(self, df):

        df = df.drop_duplicates()

        df = df.apply(
            lambda col:
            col.str.strip()
            if col.dtype == "object"
            else col
        )

        df = df.apply(
            lambda col:
            col.map(
                lambda x:
                x.lower()
                if isinstance(x, str)
                else x
            )
        )

        numeric_cols = df.select_dtypes(
            include=[np.number]
        ).columns

        df[numeric_cols] = (
            df[numeric_cols].abs()
        )

        df = df.rename(
            columns={
                "comments": "comment"
            }
        )


        excluded_columns = {
            "comment",
            "booking_id",
            "Unnamed: 0"
        }


        for col in df.columns:

            if col in excluded_columns:
                continue

            if pd.api.types.is_string_dtype(
                df[col]
            ):

                df[col] = (
                    df[col]
                    .str.strip()
                )


        # -------------------------------------------------
        # Arrival month
        # -------------------------------------------------

        if "arrival_date_month" in df.columns:

            df["arrival_date_month"] = (
                df["arrival_date_month"]
                .replace({

                    "feb": "february",
                    "apr": "april",
                    "oct": "october",
                    "nov": "november",
                    "mar": "march",
                    "jul": "july",
                    "sep": "september",
                    "dec": "december",
                    "aug": "august",
                    "jun": "june",
                    "jan": "january"

                })
            )


        # -------------------------------------------------
        # Meal
        # -------------------------------------------------

        if "meal" in df.columns:

            df["meal"] = (
                df["meal"]
                .replace({

                    "unknown": np.nan,
                    "null": np.nan,

                    "na": "SC (Self Catering)",
                    "n/a": "SC (Self Catering)",

                    "-": np.nan,
                    "": np.nan,

                    "bb": "BB (Bed & Breakfast)",
                    "hb": "HB (Half Board)",
                    "fb": "FB (Full Board)",
                    "sc": "SC (Self Catering)"

                })
            )


        # -------------------------------------------------
        # Country
        # -------------------------------------------------

        if "country" in df.columns:

            df["country"] = (
                df["country"]
                .replace({

                    "unknown": np.nan,
                    "singapore": "sgp",
                    "australia": "aus",
                    "indonesia": "idn",
                    "taiwan": "twn",
                    "china": "chn",
                    "united kingdom": "uk",
                    "south korea": "kor",
                    "philippines": "phl",
                    "hong kong": "hk",
                    "new zealand": "nzl",
                    "thailand": "tha",
                    "myanmar": "mmr",
                    "united states": "usa",
                    "vietnam": "vnm",
                    "japan": "jps",
                    "malaysia": "mys",
                    "india": "ind",
                    "hkg": "hk",
                    "-": np.nan,
                    "": np.nan

                })
            )


        # -------------------------------------------------
        # Market segment
        # -------------------------------------------------

        if "market_segment" in df.columns:

            df["market_segment"] = (
                df["market_segment"]
                .replace({

                    "unknown": np.nan,
                    "offline  ta/to": "offline-ta/to",
                    "offline  ta/t0": "offline-ta/to",
                    "offline ta/to": "offline-ta/to",
                    "offline ta/t0": "offline-ta/to",
                    "onlin ta": "online-ta",
                    "online  ta": "online-ta",
                    "online ta": "online-ta",
                    "nan": np.nan,
                    "null": np.nan,
                    "na": np.nan,
                    "-": np.nan,
                    "": np.nan

                })
            )


        # -------------------------------------------------
        # Distribution channel
        # -------------------------------------------------

        if "distribution_channel" in df.columns:

            df["distribution_channel"] = (
                df["distribution_channel"]
                .replace({

                    "unknown": np.nan,
                    "null": np.nan,
                    "-": np.nan,
                    "": np.nan

                })
            )


        # -------------------------------------------------
        # Deposit type
        # -------------------------------------------------

        if "deposit_type" in df.columns:

            df["deposit_type"] = (
                df["deposit_type"]
                .replace({

                    "no depost": "no-deposit",
                    "no  deposit": "no-deposit",
                    "no deposit": "no-deposit",
                    "non refund": "non-refund",
                    "non  refund": "non-refund",
                    "unknown": np.nan,
                    "null": np.nan,
                    "-": np.nan,
                    "": np.nan

                })
            )


        # -------------------------------------------------
        # Customer type
        # -------------------------------------------------

        if "customer_type" in df.columns:

            df["customer_type"] = (
                df["customer_type"]
                .replace({

                    "transiant": "transient",
                    "unknown": np.nan,
                    "null": np.nan,
                    "na": np.nan,
                    "n/a": np.nan,
                    "-": np.nan,
                    "": np.nan

                })
            )


        # -------------------------------------------------
        # Cancellation
        # -------------------------------------------------

        if "is_canceled" in df.columns:

            df["is_canceled"] = (
                df["is_canceled"]
                .replace({

                    "no": "0",
                    "false": "0",
                    "true": "1",
                    "yes": "1"

                })
            )


        return df


    # =====================================================
    # STEP 4.2 LEARN ENCODERS
    # =====================================================

    def _fit_encoders(self, df):

        print(
            "\n========== AUTHORITATIVE CATEGORICAL MAPPINGS =========="
        )


        for col in self.categorical_cols:

            if col not in df.columns:
                continue

            if col not in self.AUTHORITATIVE_MAPPINGS:
                continue


            self.encoders[col] = (
                self.AUTHORITATIVE_MAPPINGS[
                    col
                ].copy()
            )


            print(
                f"[INFO] Encoder fitted: {col}"
            )

            print(
                f"        {self.encoders[col]}"
            )


    # =====================================================
    # APPLY ENCODERS
    # =====================================================

    def _encode(self, df):

        df = df.copy()


        for col, mapping in self.encoders.items():

            if col not in df.columns:
                continue


            series = (
                df[col]
                .astype("string")
                .str.strip()
            )


            series = series.fillna(
                "missing"
            )


            df[col] = (
                series
                .map(mapping)
                .fillna(-1)
                .astype(int)
            )


        return df


    # =====================================================
    # PREDICT MISSING RATING
    # =====================================================

    def _predict_missing_rating(self, df):

        df = df.copy()


        if "rating" not in df.columns:

            df["rating"] = np.nan


        df["rating"] = pd.to_numeric(
            df["rating"],
            errors="coerce"
        )


        if self.rating_model is None:

            df["rating_predicted"] = np.nan

            df["final_rating"] = (
                df["rating"]
            )

            return df


        if "comment" not in df.columns:

            df["comment"] = ""


        df["comment"] = (
            df["comment"]
            .fillna("")
            .astype(str)
        )


        mask = (
            df["rating"].isna()
        )


        df["rating_predicted"] = np.nan


        if mask.sum() > 0:

            df.loc[
                mask,
                "rating_predicted"
            ] = (

                self.rating_model.predict(
                    df.loc[
                        mask,
                        "comment"
                    ]
                )

            )


        df["final_rating"] = (

            df["rating"]

            .fillna(
                df["rating_predicted"]
            )

        )


        df["final_rating"] = pd.to_numeric(
            df["final_rating"],
            errors="coerce"
        )


        df.loc[
            ~df["final_rating"].isin(
                [0, 1, 2, 3, 4, 5, 6]
            ),
            "final_rating"
        ] = np.nan


        df["final_rating"] = (
            df["final_rating"]
            .round()
            .astype("Int64")
        )


        return df


    # =====================================================
    # STEP 4.4 ITERATIVE IMPUTER
    # =====================================================

    def _fit_imputer(self, df):

        print(
            "\n========== ITERATIVE IMPUTER FIT =========="
        )


        excluded_target_cols = [

            "rating",
            "final_rating",
            "rating_predicted"

        ]


        df2 = df.drop(
            columns=excluded_target_cols,
            errors="ignore"
        ).copy()


        categorical_cols = [

            col

            for col in self.categorical_cols

            if col in df2.columns

        ]


        raw_categorical_cols = [

            col

            for col in self.raw_categorical_cols

            if col in df2.columns

        ]


        id_text_cols = [

            col

            for col in self.id_text_cols

            if col in df2.columns

        ]


        numeric_cols = [

            col

            for col in df2.columns

            if col not in categorical_cols

            and col not in raw_categorical_cols

            and col not in id_text_cols

            and not col.startswith(
                "comment_topic_"
            )
            and pd.api.types.is_numeric_dtype(
                df2[col]
            )

        ]


        # -------------------------------------------------
        # Numeric imputer
        # -------------------------------------------------

        print(
            "\n========== NUMERIC IMPUTATION =========="
        )


        self.numeric_imputer_columns = (
            numeric_cols.copy()
        )


        if numeric_cols:

            self.numeric_imputer = IterativeImputer(
                random_state=42,
                max_iter=10
            )


            self.numeric_imputer.fit(
                df2[numeric_cols]
            )


            print(
                f"[INFO] Numeric columns processed: "
                f"{len(numeric_cols)}"
            )

        else:

            self.numeric_imputer = None


        # -------------------------------------------------
        # Categorical imputer
        # -------------------------------------------------

        print(
            "\n========== CATEGORICAL IMPUTATION =========="
        )


        self.categorical_imputer_columns = (
            categorical_cols.copy()
        )


        if categorical_cols:

            self.categorical_imputer = IterativeImputer(
                random_state=42,
                max_iter=10
            )


            self.categorical_imputer.fit(
                df2[categorical_cols]
            )


            print(
                f"[INFO] Categorical columns processed: "
                f"{len(categorical_cols)}"
            )

        else:

            self.categorical_imputer = None


        self.raw_categorical_cols = (
            raw_categorical_cols
        )

        self.id_text_cols = (
            id_text_cols
        )


        print(
            "\n========== IMPUTER COLUMN SUMMARY =========="
        )

        print(
            f"[INFO] Numeric columns: "
            f"{len(self.numeric_imputer_columns)}"
        )

        print(
            f"[INFO] Encoded categorical columns: "
            f"{len(self.categorical_imputer_columns)}"
        )

        print(
            f"[INFO] Raw categorical/text columns: "
            f"{len(self.raw_categorical_cols)}"
        )

        print(
            f"[INFO] ID/text columns: "
            f"{len(self.id_text_cols)}"
        )

        print(
            "[INFO] comment_topic_* columns excluded "
            "from IterativeImputer."
        )

        print(
            "[INFO] rating, final_rating and "
            "rating_predicted excluded from "
            "IterativeImputer."
        )


    # =====================================================
    # NORMAL ITERATIVE IMPUTATION
    # =====================================================

    def _impute(self, df):

        df = df.copy()


        # -------------------------------------------------
        # Numeric
        # -------------------------------------------------

        if (

            self.numeric_imputer is not None

            and

            self.numeric_imputer_columns

        ):

            numeric_cols = [

                col

                for col
                in self.numeric_imputer_columns

                if col in df.columns

            ]


            if numeric_cols:

                df[numeric_cols] = (

                    self.numeric_imputer.transform(
                        df[numeric_cols]
                    )

                )


        # -------------------------------------------------
        # Categorical
        # -------------------------------------------------

        if (

            self.categorical_imputer is not None

            and

            self.categorical_imputer_columns

        ):

            categorical_cols = [

                col

                for col
                in self.categorical_imputer_columns

                if col in df.columns

            ]


            if categorical_cols:

                df[categorical_cols] = (

                    self.categorical_imputer.transform(
                        df[categorical_cols]
                    )

                )


                for col in categorical_cols:

                    if col not in self.categorical_ranges:
                        continue


                    minimum, maximum = (
                        self.categorical_ranges[col]
                    )


                    df[col] = (

                        df[col]
                        .round()
                        .clip(
                            lower=minimum,
                            upper=maximum
                        )
                        .astype(int)

                    )


        # -------------------------------------------------
        # Raw categorical / text
        # -------------------------------------------------

        print(
            "\n========== RAW CATEGORICAL IMPUTATION =========="
        )


        for col in self.raw_categorical_cols:

            if col not in df.columns:
                continue


            missing_before = (
                df[col].isna().sum()
            )


            if missing_before > 0:

                df[col] = (
                    df[col]
                    .fillna("nan")
                )


                print(
                    f"[INFO] {col}: "
                    f"{missing_before:,} missing values "
                    f"filled with 'nan'"
                )


        # -------------------------------------------------
        # Verify
        # -------------------------------------------------

        print(
            "\n========== AFTER IMPUTATION =========="
        )


        missing_after = (
            df.isna().sum()
        )


        missing_display = missing_after[
            ~missing_after.index.str.startswith(
                "comment_topic_"
            )
        ]


        print(
            "\nMissing values:"
        )

        print(
            missing_display.to_string()
        )


        remaining_missing = (
            missing_display[
                missing_display > 0
            ]
        )


        print(
            "\n========== REMAINING MISSING VALUES =========="
        )


        if len(remaining_missing) == 0:

            print(
                "No missing values."
            )

        else:

            print(
                remaining_missing.to_string()
            )


        # -------------------------------------------------
        # Reservation status
        # -------------------------------------------------

        if "reservation_status" in df.columns:

            print(
                "\n========== RESERVATION STATUS =========="
            )


            status_counts = (
                df["reservation_status"]
                .value_counts()
                .sort_index()
            )


            print(
                status_counts
            )


            print(
                "\nReservation Status Percentages:"
            )


            print(

                df["reservation_status"]
                .value_counts(
                    normalize=True
                )
                .sort_index()
                .mul(100)
                .round(2)

            )


            print(
                "\nReservation Status Labels:"
            )


            status_labels = (
                df["reservation_status"]
                .map(
                    self.REVERSE_MAPPINGS[
                        "reservation_status"
                    ]
                )
            )


            print(
                status_labels.value_counts()
            )


        print(
            "DataProcessor - rating columns "
            "kept separate from IterativeImputer"
        )

        print(
            "DataProcessor - final_rating preserved "
            "for downstream ML processing"
        )


        return df





