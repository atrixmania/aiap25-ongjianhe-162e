# =========================
# src/preprocess.py
# =========================

import numpy as np
import pandas as pd
import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning
)


class DataProcessor:

    REVERSE_MAPPINGS = {
        'hotel': {
            0: 'city hotel',
            1: 'airport hotel'
        },

        'meal': {
            0: 'BB (Bed & Breakfast)',
            1: 'HB (Half Board)',
            2: 'FB (Full Board)',
            3: 'SC (Self Catering)',
            4: 'undefined',
            5: np.nan
        },

        'market_segment': {
            0: 'offline-ta/to',
            1: 'online-ta',
            2: 'groups',
            3: 'direct',
            4: 'corporate',
            5: 'complementary',
            6: 'aviation',
            7: 'undefined',
            8: np.nan
        },

        'distribution_channel': {
            0: 'direct',
            1: 'corporate',
            2: 'gds',
            3: 'undefined',
            4: 'ta/to',
            5: np.nan
        },

        'deposit_type': {
            0: 'non-refund',
            1: 'no-deposit',
            2: 'refundable'
        },

        'customer_type': {
            0: 'transient',
            1: 'transient-party',
            2: 'contract',
            3: 'group',
            4: np.nan
        },

        'reservation_status': {
            0: 'canceled',
            1: 'no-show',
            2: 'check-out'
        }
    }


    def __init__(self, config=None):

        self.config = config or {}

        # Normal categorical encoders
        self.encoders = {}

        # -------------------------------------------------
        # Rating model
        # -------------------------------------------------

        self.rating_model = None

        self.rating_classes = []


        # -------------------------------------------------
        # Normal IterativeImputer
        # -------------------------------------------------

        self.imputer = None

        self.imputer_encoders = {}

        self.imputer_columns = []


    # =====================================================
    # PREPARE FINAL RATING
    # =====================================================
    #
    # IMPORTANT:
    #
    # This method is SEPARATE from IterativeImputer.
    #
    # It performs:
    #
    #     rating
    #        ↓
    #     comment rating model
    #        ↓
    #     predicted missing rating
    #        ↓
    #     final_rating
    #
    # final_rating becomes the ML TARGET.
    #
    # It is NOT a normal feature imputation.
    #
    # =====================================================

    def prepare_final_rating(self, df):

        df = df.copy()


        # -------------------------------------------------
        # Clean data first
        # -------------------------------------------------

        df = self._clean_data(
            df
        )


        # -------------------------------------------------
        # Ensure rating is numeric
        # -------------------------------------------------

        if "rating" not in df.columns:

            raise KeyError(
                "Column 'rating' is required "
                "to create final_rating."
            )


        df["rating"] = pd.to_numeric(
            df["rating"],
            errors="coerce"
        )


        # -------------------------------------------------
        # Ensure rating is within 0-6
        # -------------------------------------------------

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


        # =================================================
        # FIT RATING MODEL
        # =================================================

        train = df[
            df["rating"].notna()
        ].copy()


        if len(train) == 0:

            raise ValueError(
                "No valid ratings are available "
                "to train the rating model."
            )


        X = train[
            "comment"
        ]

        y = train[
            "rating"
        ].astype(int)


        # -------------------------------------------------
        # Store actual rating classes
        # -------------------------------------------------

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


        # =================================================
        # PREDICT MISSING RATINGS
        # =================================================

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


        # =================================================
        # CREATE FINAL RATING
        # =================================================

        df["final_rating"] = (

            df["rating"]

            .fillna(
                df["rating_predicted"]
            )
        )


        # -------------------------------------------------
        # Ensure final_rating is integer 0-6
        # -------------------------------------------------

        df["final_rating"] = pd.to_numeric(
            df["final_rating"],
            errors="coerce"
        )


        # Safety protection
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


        # =================================================
        # LOG RESULTS
        # =================================================

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


        # -------------------------------------------------
        # Clean
        # -------------------------------------------------

        df = self._clean_data(
            df
        )


        # -------------------------------------------------
        # Fit categorical encoders
        # -------------------------------------------------

        self._fit_encoders(
            df
        )


        # -------------------------------------------------
        # Encode
        # -------------------------------------------------

        df = self._encode(
            df
        )


        # -------------------------------------------------
        # IMPORTANT
        #
        # Do NOT fit rating model here.
        #
        # final_rating has already been prepared BEFORE
        # the train/validation/test split.
        #
        # -------------------------------------------------


        # -------------------------------------------------
        # Fit normal IterativeImputer
        # -------------------------------------------------

        self._fit_imputer(
            df
        )


        print(
            "DataProcessor fitted"
        )


        return self


    # =====================================================
    # TRANSFORM
    # =====================================================

    def transform(self, df):

        df = df.copy()


        # -------------------------------------------------
        # Clean
        # -------------------------------------------------

        df = self._clean_data(
            df
        )


        # -------------------------------------------------
        # Encode
        # -------------------------------------------------

        df = self._encode(
            df
        )


        # -------------------------------------------------
        # Create final_rating when processing RAW data
        #
        # This is useful for dashboard/prediction data.
        #
        # If final_rating already exists, do not overwrite it.
        # -------------------------------------------------

        if "final_rating" not in df.columns:

            df = self._predict_missing_rating(
                df
            )


        # -------------------------------------------------
        # Normal feature imputation
        # -------------------------------------------------

        df = self._impute(
            df
        )


        print(
            "DataProcessor transform completed"
        )


        return df


    # =====================================================
    # FIT TRANSFORM
    # =====================================================

    def fit_transform(self, train_df):

        self.fit(
            train_df
        )

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


        # -------------------------------------------------
        # Columns that must NEVER be touched
        # -------------------------------------------------

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

        cols = [
            "hotel",
            "meal",
            "market_segment",
            "distribution_channel",
            "deposit_type",
            "customer_type",
            "reservation_status"
        ]


        for col in cols:

            if col not in df.columns:
                continue


            values = (
                df[col]
                .astype("string")
                .fillna("missing")
                .unique()
            )


            self.encoders[col] = {

                value: index

                for index, value
                in enumerate(
                    sorted(values)
                )
            }


    # =====================================================
    # APPLY ENCODERS
    # =====================================================

    def _encode(self, df):

        for col, mapping in self.encoders.items():

            if col not in df.columns:
                continue


            df[col] = (

                df[col]

                .astype("string")

                .fillna("missing")

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


        # -------------------------------------------------
        # Ensure rating exists
        # -------------------------------------------------

        if "rating" not in df.columns:

            df["rating"] = np.nan


        df["rating"] = pd.to_numeric(
            df["rating"],
            errors="coerce"
        )


        # -------------------------------------------------
        # If rating model doesn't exist
        # -------------------------------------------------

        if self.rating_model is None:

            df["rating_predicted"] = np.nan

            df["final_rating"] = (
                df["rating"]
            )

            return df


        # -------------------------------------------------
        # Prepare comments
        # -------------------------------------------------

        if "comment" not in df.columns:

            df["comment"] = ""


        df["comment"] = (
            df["comment"]
            .fillna("")
            .astype(str)
        )


        # -------------------------------------------------
        # Find missing ratings
        # -------------------------------------------------

        mask = (
            df["rating"].isna()
        )


        df["rating_predicted"] = np.nan


        # -------------------------------------------------
        # Predict missing ratings
        # -------------------------------------------------

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


        # -------------------------------------------------
        # Create final_rating
        # -------------------------------------------------

        df["final_rating"] = (

            df["rating"]

            .fillna(
                df["rating_predicted"]
            )
        )


        # -------------------------------------------------
        # Force 0-6 integer target
        # -------------------------------------------------

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
    # STEP 4.4 NORMAL ITERATIVE IMPUTER
    # =====================================================

    def _fit_imputer(self, df):

        # -------------------------------------------------
        # IMPORTANT
        #
        # These are TARGET/RATING columns.
        #
        # IterativeImputer must NEVER touch them.
        # -------------------------------------------------

        df2 = df.drop(
            columns=[
                "rating",
                "final_rating",
                "rating_predicted"
            ],
            errors="ignore"
        ).copy()


        # -------------------------------------------------
        # Categorical columns
        # -------------------------------------------------

        cat_cols = df2.select_dtypes(
            include="object"
        ).columns


        for col in cat_cols:

            le = LabelEncoder()


            df2[col] = (
                le.fit_transform(
                    df2[col].astype(str)
                )
            )


            self.imputer_encoders[col] = le


        # -------------------------------------------------
        # Store column order
        # -------------------------------------------------

        self.imputer_columns = (
            df2.columns.tolist()
        )


        # -------------------------------------------------
        # IterativeImputer
        # -------------------------------------------------

        self.imputer = IterativeImputer(
            random_state=42,
            max_iter=10
        )


        self.imputer.fit(
            df2
        )


    # =====================================================
    # NORMAL ITERATIVE IMPUTATION
    # =====================================================

    def _impute(self, df):

        if self.imputer is None:

            return df


        df2 = df.drop(
            columns=[
                "rating",
                "final_rating",
                "rating_predicted"
            ],
            errors="ignore"
        ).copy()


        # -------------------------------------------------
        # Make sure all training columns exist
        # -------------------------------------------------

        for col in self.imputer_columns:

            if col not in df2.columns:

                df2[col] = np.nan


        # -------------------------------------------------
        # Remove unexpected columns
        # -------------------------------------------------

        df2 = df2[
            self.imputer_columns
        ].copy()


        # =================================================
        # Encode categorical columns
        # =================================================

        for col, le in self.imputer_encoders.items():

            mapping = {

                value: index

                for index, value
                in enumerate(
                    le.classes_
                )
            }


            df2[col] = (

                df2[col]

                .astype(str)

                .map(mapping)

                .fillna(-1)

                .astype(int)
            )


        # =================================================
        # Run IterativeImputer
        # =================================================

        arr = self.imputer.transform(
            df2
        )


        result = pd.DataFrame(

            arr,

            columns=self.imputer_columns,

            index=df.index
        )


        # =================================================
        # Restore categorical values
        # =================================================

        for col, le in self.imputer_encoders.items():

            encoded_values = (

                result[col]

                .round()

                .astype(int)
            )


            encoded_values = (

                encoded_values

                .clip(
                    lower=0,
                    upper=len(le.classes_) - 1
                )
            )


            result[col] = (

                le.inverse_transform(
                    encoded_values
                )
            )


        # =================================================
        # Restore ORIGINAL rating
        # =================================================

        if "rating" in df.columns:

            result["rating"] = (
                df["rating"]
            )


        # =================================================
        # Restore final_rating
        # =================================================

        if "final_rating" in df.columns:

            result["final_rating"] = (
                df["final_rating"]
            )


        # =================================================
        # Restore rating_predicted
        # =================================================

        if "rating_predicted" in df.columns:

            result["rating_predicted"] = (
                df["rating_predicted"]
            )


        print(
            "DataProcessor - rating columns "
            "kept separate from IterativeImputer"
        )


        return result



