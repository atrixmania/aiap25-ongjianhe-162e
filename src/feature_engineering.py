# =========================
# src/feature_engineering.py
# =========================

import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

from sklearn.utils.validation import check_is_fitted

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning
)


class FeatureEngineer:

    def __init__(self, config=None):

        self.config = config or {}

        self.client_stats = None

        self.client_bad_stats = None

        self.top_bad_clients = []

        self.comment_model = None
        self.comment_cluster_model = None
        self.comment_categories = {}

        self.comment_embedding_model = None

        self.comment_cluster_centroids = {}

        self.comment_cluster_samples = {}

        self.comment_cluster_labels = {}

        # =====================================================
        # Step 5.6 COMMENT CLUSTER CACHE
        #
        # cleaned comment -> cluster ID
        #
        # This prevents the same comment from being embedded
        # repeatedly during FIT / TRANSFORM.
        # =====================================================

        self.comment_cluster_cache = {}

        self.top20_comments = []


    # =====================================================
    # FIT
    # =====================================================

    def fit(self, train_df):

        df = train_df.copy()

        df = self._business_rules(df)

        df = self._lead_time(df)

        self._fit_clientdriver_behaviour(df)

        self._fit_bad_clientdriver(df)

        df = self._apply_clientdriver_behaviour(df)

        df = self._apply_bad_clientdriver(df)

        df = self._top_comment(df)

        self._fit_top20_comment(df)

        df = self._apply_top20_comment(df)

        print(
            "FeatureEngineering fitted"
        )

        return self


    # =====================================================
    # TRANSFORM
    # =====================================================

    def transform(self, df):

        if self.client_stats is None:

            raise RuntimeError(
                "DataProcessor must be fitted before transform()"
            )

        df = df.copy()

        df = self._business_rules(df)

        df = self._lead_time(df)

        df = self._apply_clientdriver_behaviour(df)

        df = self._apply_bad_clientdriver(df)

        df = self._top_comment(df)

        df = self._apply_top20_comment(df)

        print(
            "FeatureEngineering transform completed"
        )

        return df


    # =====================================================
    # FIT TRANSFORM
    # =====================================================

    def fit_transform(self, train_df):

        self.fit(train_df)

        return self.transform(train_df)


    # =========================================================
    # Step 5.1 Feature Engineering (Business Rules)
    # =========================================================

    def _business_rules(self, df):

        # Convert final rating to numeric
        df["final_rating"] = pd.to_numeric(
            df["final_rating"],
            errors="coerce"
        )

        df["season_group"] = np.select(
            [
                df["arrival_date_month"].str.lower().isin([
                    "june",
                    "july",
                    "august",
                    "december"
                ]),

                df["arrival_date_month"].str.lower().isin([
                    "january",
                    "february",
                    "march",
                    "april",
                    "may",
                    "september",
                    "october",
                    "november"
                ])
            ],

            [
                "peak_season",
                "off_peak_season"
            ],

            default="others"
        )


        df["country_group"] = np.select(
            [
                df["country"].str.lower().isin([
                    "sgp"
                ]),

                df["country"].str.lower().isin([
                    "idn",
                    "ind",
                    "phl",
                    "mys",
                    "tha",
                    "mmr",
                    "vnm"
                ]),

                df["country"].str.lower().isin([
                    "chn",
                    "hkg",
                    "twn",
                    "jpn",
                    "hk",
                    "kor",
                ]),

                df["country"].str.lower().isin([
                    "aus",
                    "nzl",
                ]),

                df["country"].str.lower().isin([
                    "usa",
                ]),

                df["country"].str.lower().isin([
                    "gbr",
                    "uk"
                ]),
            ],

            [
                "local",
                "southeast_asia",
                "north_asia",
                "oceania",
                "America",
                "Europe"
            ],

            default="others"
        )


        df["adr"] = pd.to_numeric(
            df["adr"],
            errors="coerce"
        )


        df["adr_group"] = np.select(
            [
                df["adr"] <= 100,

                (
                    (df["adr"] > 100) &
                    (df["adr"] <= 500)
                ),

                (
                    (df["adr"] > 500) &
                    (df["adr"] <= 1000)
                ),

                df["adr"] > 1000
            ],

            [
                "Low value (≤100)",
                "Mid value (100–500)",
                "High value (500–1k)",
                "Premium (1K+)"
            ],

            default="Unknown"
        )


        df["previous_cancellations"] = pd.to_numeric(
            df["previous_cancellations"],
            errors="coerce"
        )


        df["previous_cancellations_group"] = np.select(
            [
                df["previous_cancellations"] <= 3,

                (
                    (df["previous_cancellations"] > 3) &
                    (df["previous_cancellations"] <= 7)
                ),

                (
                    (df["previous_cancellations"] > 7) &
                    (df["previous_cancellations"] <= 10)
                ),

                df["previous_cancellations"] > 10
            ],

            [
                "Low cancellation (≤3)",
                "Medium cancellation (4–7)",
                "High cancellation (7–10)",
                "Very high cancellation (10+)"
            ],

            default="Unknown"
        )


        df["days_in_waiting_list"] = pd.to_numeric(
            df["days_in_waiting_list"],
            errors="coerce"
        )


        df["days_in_waiting_list_group"] = np.select(
            [
                df["days_in_waiting_list"] <= 7,

                (
                    (df["days_in_waiting_list"] > 7) &
                    (df["days_in_waiting_list"] <= 14)
                ),

                (
                    (df["days_in_waiting_list"] > 14) &
                    (df["days_in_waiting_list"] <= 30)
                ),

                df["days_in_waiting_list"] > 30
            ],

            [
                "Short (0-7 days)",
                "Mid (8–14 days)",
                "Long (15–30 days)",
                "Extreme Long (30+ days)"
            ],

            default="Unknown"
        )


        print(
            "Pre-process - Business Rules by category"
        )

        return df


    # =========================================================
    # Step 5 Feature Engineering 2
    # Lead time computation
    # =========================================================

    def _lead_time(self, df):

        # =====================================================
        # Family_size computation
        # =====================================================

        df["adults"] = pd.to_numeric(
            df["adults"],
            errors="coerce"
        )

        df["children"] = pd.to_numeric(
            df["children"],
            errors="coerce"
        )

        df["babies"] = pd.to_numeric(
            df["babies"],
            errors="coerce"
        )


        df["total_guests"] = (
            df["adults"].fillna(0)
            +
            df["children"].fillna(0)
            +
            df["babies"].fillna(0)
        )


        df["guest_group"] = np.select(
            [
                (
                    (df["adults"] == 1) &
                    (df["children"] == 0) &
                    (df["babies"] == 0)
                ),

                (
                    (df["adults"] == 2) &
                    (df["children"] == 0) &
                    (df["babies"] == 0)
                ),

                (
                    (df["children"] > 0) |
                    (df["babies"] > 0)
                ),

                (
                    (df["adults"] >= 3) &
                    (df["children"] == 0) &
                    (df["babies"] == 0)
                )
            ],

            [
                "Single",
                "Couple",
                "Family",
                "Group"
            ],

            default="Unknown"
        )


        print(
            "Pre-process - Lead time & Service Level Computation"
        )

        return df


    # =========================================================
    # Step 5.3 Client & Driver Behaviour - FIT
    # =========================================================

    def _fit_clientdriver_behaviour(self, df):

        self.client_stats = (
            df.groupby("client_email")["final_rating"]
            .mean()
            .reset_index()
        )


        self.client_stats.columns = [
            "client_email",
            "client_avg_rating"
        ]


        def client_label(r):

            if r <= 2:

                return "Client - bad review"

            elif r == 3:

                return "Client - neutral review"

            else:

                return "Client - good review"


        self.client_stats["client_segment"] = (
            self.client_stats["client_avg_rating"]
            .apply(client_label)
        )


        print(
            "Feature Engineering - "
            "Client & Driver Statistics fitted"
        )


    # =========================================================
    # Step 5.3 Client & Driver Behaviour - TRANSFORM
    # =========================================================

    def _apply_clientdriver_behaviour(self, df):

        df = df.merge(
            self.client_stats[
                [
                    "client_email",
                    "client_segment"
                ]
            ],

            on="client_email",

            how="left"
        )


        df["client_segment"] = (
            df["client_segment"]
            .fillna("Client - unknown")
        )


        return df


    # =========================================================
    # Step 5.4 Bad Client & Driver - FIT
    # =========================================================

    def _fit_bad_clientdriver(self, df):

        df = df.copy()


        df["is_bad_rating"] = (
            df["final_rating"]
            .isin([0, 1, 2])
            .astype(int)
        )


        self.client_bad_stats = (
            df.groupby("client_email")
            .agg(
                total=("final_rating", "count"),
                bad_count=("is_bad_rating", "sum")
            )
            .reset_index()
        )


        self.client_bad_stats["client_bad_rate"] = (
            self.client_bad_stats["bad_count"]
            /
            self.client_bad_stats["total"]
        )


        top_10_bad_clients = (
            self.client_bad_stats
            .sort_values(
                "client_bad_rate",
                ascending=False
            )
            .head(10)
        )


        self.top_bad_clients = (
            top_10_bad_clients["client_email"]
            .tolist()
        )


        print(
            "Feature Engineering - "
            "Bad Client Driver Statistics fitted"
        )


    # =========================================================
    # Step 5.4 Bad Client & Driver - TRANSFORM
    # =========================================================

    def _apply_bad_clientdriver(self, df):

        df = df.copy()


        df = df.merge(
            self.client_bad_stats[
                [
                    "client_email",
                    "client_bad_rate"
                ]
            ],

            on="client_email",

            how="left"
        )


        df["client_bad_rate"] = (
            df["client_bad_rate"]
            .fillna(0)
        )


        df["is_top_bad_client"] = (
            df["client_email"]
            .isin(self.top_bad_clients)
            .astype(int)
        )


        df.attrs["top_10_bad_clients"] = (
            self.top_bad_clients
        )


        return df


    # =========================================================
    # Step 5.5 Top Comment Memory Store
    # =========================================================

    def _top_comment(self, df):

        from collections import defaultdict

        df = df.copy()


        df["comment"] = (
            df["comment"]
            .fillna("")
            .astype(str)
        )


        # =====================================================
        # RATING LEVEL
        # =====================================================

        rating_db = (
            df.groupby(
                "final_rating",
                dropna=False
            )["comment"]
            .value_counts()
            .groupby(level=0)
            .head(10)
            .groupby(level=0)
            .apply(
                lambda x:
                x.index.get_level_values(1).tolist()
            )
            .to_dict()
        )


        # =====================================================
        # CLIENT LEVEL
        # =====================================================

        client_db = (
            df.groupby(
                "client_email",
                dropna=True
            )["comment"]
            .value_counts()
            .groupby(level=0)
            .head(10)
            .groupby(level=0)
            .apply(
                lambda x:
                x.index.get_level_values(1).tolist()
            )
            .to_dict()
        )


        # =====================================================
        # STORE
        # =====================================================

        df.attrs["top_comments_rating"] = (
            rating_db
        )

        df.attrs["top_comments_client"] = (
            client_db
        )


        print(
            "Pre-process - "
            "Step 5.5 Top Comments Memory Built"
        )


        return df


    # =========================================================
    # Step 5.6 Top Comment Clustering - FIT
    # =========================================================

    def _fit_top20_comment(self, df):

        import re

        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import AgglomerativeClustering


        print(
            "Feature Engineering - "
            "Step 5.6 FIT Comment Clustering"
        )


        # =====================================================
        # 1. ONLY BAD REVIEWS
        # =====================================================

        df_bad = df[
            df["final_rating"].isin([0, 1, 2])
        ].copy()


        # =====================================================
        # 2. CLEAN COMMENTS
        # =====================================================

        df_bad["comment"] = (
            df_bad["comment"]
            .fillna("")
            .astype(str)
            .str.lower()
        )


        def clean_text(text):

            text = re.sub(
                r'ðÿ\S*',
                ' ',
                text
            )

            text = re.sub(
                r'[^a-z0-9\s]',
                ' ',
                text
            )

            text = re.sub(
                r'\s+',
                ' ',
                text
            )

            return text.strip()


        df_bad["comment_clean"] = (
            df_bad["comment"]
            .apply(clean_text)
        )


        # =====================================================
        # 3. REMOVE EMPTY COMMENTS
        # =====================================================

        df_bad = df_bad[
            df_bad["comment_clean"].str.len() > 0
        ].copy()


        # =====================================================
        # 4. COUNT COMMENT FREQUENCY
        # =====================================================

        comment_counts = (
            df_bad["comment_clean"]
            .value_counts()
        )


        # =====================================================
        # 5. TOP 20 COMMENTS
        # =====================================================

        top20_comments = (
            comment_counts
            .head(20)
        )


        if top20_comments.empty:

            print(
                "Step 5.6 - "
                "No valid bad-review comments found"
            )


            self.comment_embedding_model = None

            self.comment_cluster_centroids = {}

            self.comment_cluster_samples = {}

            self.comment_cluster_labels = {}

            self.comment_cluster_cache = {}

            self.top20_comments = []

            return


        self.top20_comments = (
            top20_comments.index.tolist()
        )


        print(
            f"Step 5.6 - Top "
            f"{len(self.top20_comments)} comments selected"
        )


        for i, comment in enumerate(
            self.top20_comments,
            start=1
        ):

            print(
                f"  {i:02d}. "
                f"{comment} "
                f"(count={top20_comments[comment]})"
            )


        # =====================================================
        # 6. SENTENCE TRANSFORMER
        # =====================================================

        print(
            "Step 5.6 - Loading SentenceTransformer..."
        )


        self.comment_embedding_model = (
            SentenceTransformer(
                "all-MiniLM-L6-v2"
            )
        )


        # =====================================================
        # 7. EMBED TOP 20 COMMENTS ONLY
        # =====================================================

        embeddings = (
            self.comment_embedding_model.encode(
                self.top20_comments,

                convert_to_numpy=True,

                normalize_embeddings=True,

                batch_size=20,

                show_progress_bar=False
            )
        )


        print(
            f"Step 5.6 - Generated embeddings for "
            f"{len(self.top20_comments)} comments"
        )


        # =====================================================
        # 8. CREATE 10 CLUSTERS
        # =====================================================

        n_clusters = min(
            10,
            len(self.top20_comments)
        )


        if n_clusters == 1:

            clusters = np.zeros(
                len(self.top20_comments),
                dtype=int
            )

        else:

            cluster_model = (
                AgglomerativeClustering(
                    n_clusters=n_clusters,
                    metric="cosine",
                    linkage="average"
                )
            )


            clusters = (
                cluster_model.fit_predict(
                    embeddings
                )
            )


        # =====================================================
        # 9. STORE CLUSTER LABELS
        # =====================================================

        self.comment_cluster_labels = dict(
            zip(
                self.top20_comments,
                clusters
            )
        )


        # =====================================================
        # 10. INITIALIZE CACHE
        #
        # The top 20 comments already have their cluster.
        # Therefore they never need to be embedded again.
        # =====================================================

        self.comment_cluster_cache = dict(
            self.comment_cluster_labels
        )


        # =====================================================
        # 11. STORE CLUSTER CENTROIDS
        # =====================================================

        self.comment_cluster_centroids = {}

        self.comment_cluster_samples = {}


        for cluster_id in sorted(
            set(clusters)
        ):

            cluster_mask = (
                clusters == cluster_id
            )


            cluster_embeddings = (
                embeddings[
                    cluster_mask
                ]
            )


            # =================================================
            # Mean embedding
            # =================================================

            centroid = (
                cluster_embeddings
                .mean(axis=0)
            )


            # =================================================
            # Normalize centroid
            # =================================================

            centroid_norm = np.linalg.norm(
                centroid
            )


            if centroid_norm > 0:

                centroid = (
                    centroid
                    /
                    centroid_norm
                )


            self.comment_cluster_centroids[
                cluster_id
            ] = centroid


            # =================================================
            # Store sample comments
            # =================================================

            cluster_comments = [

                self.top20_comments[i]

                for i in range(
                    len(self.top20_comments)
                )

                if clusters[i] == cluster_id
            ]


            self.comment_cluster_samples[
                cluster_id
            ] = cluster_comments[:3]


        print(
            "Feature Engineering - "
            "Step 5.6 FIT Top Comment Clustering completed"
        )


        print(
            f"Step 5.6 - "
            f"{len(self.comment_cluster_centroids)} "
            f"clusters created"
        )


    # =========================================================
    # Step 5.6 Top Comment Clustering - APPLY
    # =========================================================

    def _apply_top20_comment(self, df):

        import re

        print(
            "Feature Engineering - "
            "Step 5.6 APPLY Comment Clustering"
        )


        df = df.copy()


        # =====================================================
        # 1. CLEAN COMMENTS
        # =====================================================

        df["comment"] = (
            df["comment"]
            .fillna("")
            .astype(str)
            .str.lower()
        )


        def clean_text(text):

            text = re.sub(
                r'ðÿ\S*',
                ' ',
                text
            )

            text = re.sub(
                r'[^a-z0-9\s]',
                ' ',
                text
            )

            text = re.sub(
                r'\s+',
                ' ',
                text
            )

            return text.strip()


        df["comment_clean"] = (
            df["comment"]
            .apply(clean_text)
        )


        # =====================================================
        # 2. CHECK WHETHER MODEL EXISTS
        # =====================================================

        if (
            self.comment_embedding_model is None
            or not self.comment_cluster_centroids
        ):

            df["cluster"] = -1

            df["business_categories"] = (
                "Other / Acceptable Stay"
            )

            df["sample_comments"] = None

            return df


        # =====================================================
        # 3. ENSURE CACHE EXISTS
        #
        # IMPORTANT:
        #
        # Older feature_engineering.pkl files may have been
        # created before comment_cluster_cache existed.
        #
        # This prevents:
        #
        # AttributeError:
        # 'FeatureEngineer' object has no attribute
        # 'comment_cluster_cache'
        # =====================================================

        if not hasattr(
            self,
            "comment_cluster_cache"
        ):

            self.comment_cluster_cache = {}


        # =====================================================
        # 4. RESTORE CACHE FROM CLUSTER LABELS IF POSSIBLE
        # =====================================================

        if (
            not self.comment_cluster_cache
            and getattr(
                self,
                "comment_cluster_labels",
                None
            )
        ):

            self.comment_cluster_cache = dict(
                self.comment_cluster_labels
            )


        # =====================================================
        # 5. UNIQUE NON-EMPTY COMMENTS ONLY
        # =====================================================

        unique_comments = (
            df.loc[
                df["comment_clean"].str.len() > 0,
                "comment_clean"
            ]
            .drop_duplicates()
            .tolist()
        )


        if not unique_comments:

            df["cluster"] = -1

            df["business_categories"] = (
                "Other / Acceptable Stay"
            )

            df["sample_comments"] = None

            return df


        print(
            f"Step 5.6 - Applying clustering to "
            f"{len(unique_comments):,} unique comments "
            f"from {len(df):,} rows"
        )


        # =====================================================
        # 6. FIND COMMENTS NOT IN CACHE
        # =====================================================

        uncached_comments = [

            comment

            for comment in unique_comments

            if comment not in self.comment_cluster_cache
        ]


        cached_count = (
            len(unique_comments)
            -
            len(uncached_comments)
        )


        print(
            f"Step 5.6 - Cached comments: "
            f"{cached_count:,}"
        )


        print(
            f"Step 5.6 - New comments requiring embedding: "
            f"{len(uncached_comments):,}"
        )


        # =====================================================
        # 7. EMBED ONLY NEW COMMENTS
        # =====================================================

        if uncached_comments:

            embeddings = (
                self.comment_embedding_model.encode(

                    uncached_comments,

                    convert_to_numpy=True,

                    normalize_embeddings=True,

                    batch_size=64,

                    show_progress_bar=False
                )
            )


            # =================================================
            # 8. BUILD CENTROID MATRIX
            # =================================================

            cluster_ids = list(
                self.comment_cluster_centroids.keys()
            )


            centroid_matrix = np.vstack(
                [
                    self.comment_cluster_centroids[
                        cluster_id
                    ]

                    for cluster_id in cluster_ids
                ]
            )


            # =================================================
            # 9. COSINE SIMILARITY
            #
            # Normalized vectors:
            #
            # cosine similarity = dot product
            # =================================================

            similarity_matrix = (
                embeddings
                @
                centroid_matrix.T
            )


            # =================================================
            # 10. FIND BEST CLUSTER
            # =================================================

            best_cluster_indices = (
                np.argmax(
                    similarity_matrix,
                    axis=1
                )
            )


            unique_cluster_labels = [

                cluster_ids[index]

                for index
                in best_cluster_indices
            ]


            # =================================================
            # 11. ADD NEW RESULTS TO CACHE
            # =================================================

            for comment, cluster_id in zip(
                uncached_comments,
                unique_cluster_labels
            ):

                self.comment_cluster_cache[
                    comment
                ] = cluster_id


        # =====================================================
        # 12. COMMENT -> CLUSTER LOOKUP
        # =====================================================

        df["cluster"] = (
            df["comment_clean"]
            .map(
                self.comment_cluster_cache
            )
            .fillna(-1)
            .astype(int)
        )


        # =====================================================
        # 13. BUSINESS CATEGORIES
        # =====================================================

        def extract_business(text):

            labels = []


            # =================================================
            # Room Cleanliness
            # =================================================

            if any(
                keyword in text

                for keyword in [

                    "dirty",
                    "unclean",
                    "not clean",
                    "cleanliness",
                    "dust",
                    "dusty",
                    "stain",
                    "stained",
                    "smell",
                    "smelly",
                    "bad smell",
                    "odor",
                    "odour",
                    "mold",
                    "mould",
                    "hair",
                    "bed sheet",
                    "bedsheet",
                    "towel",
                    "toilet dirty",
                    "bathroom dirty"
                ]
            ):

                labels.append(
                    "Room Cleanliness Issue"
                )


            # =================================================
            # Room Condition / Maintenance
            # =================================================

            if any(
                keyword in text

                for keyword in [

                    "broken",
                    "damaged",
                    "not working",
                    "does not work",
                    "did not work",
                    "air conditioning",
                    "aircon",
                    "a c",
                    "ac not",
                    "shower",
                    "water pressure",
                    "hot water",
                    "toilet",
                    "bathroom",
                    "leak",
                    "leaking",
                    "maintenance",
                    "light not",
                    "tv not",
                    "television not",
                    "fridge not",
                    "refrigerator not",
                    "door lock",
                    "lock not",
                    "lift not",
                    "elevator not"
                ]
            ):

                labels.append(
                    "Room Condition / Maintenance Issue"
                )


            # =================================================
            # Staff / Service
            # =================================================

            if any(
                keyword in text

                for keyword in [

                    "staff",
                    "reception",
                    "service",
                    "customer service",
                    "rude",
                    "unfriendly",
                    "unhelpful",
                    "unprofessional",
                    "attitude",
                    "helpful",
                    "waiter",
                    "manager",
                    "front desk"
                ]
            ):

                labels.append(
                    "Staff / Service Issue"
                )


            # =================================================
            # Check-in / Check-out
            # =================================================

            if any(
                keyword in text

                for keyword in [

                    "check in",
                    "check-in",
                    "checkin",
                    "check out",
                    "check-out",
                    "checkout",
                    "waiting to check",
                    "long wait",
                    "waited",
                    "queue",
                    "early check",
                    "late check"
                ]
            ):

                labels.append(
                    "Check-in / Check-out Issue"
                )


            # =================================================
            # Booking / Reservation
            # =================================================

            if any(
                keyword in text

                for keyword in [

                    "booking",
                    "booked",
                    "reservation",
                    "reserved",
                    "reservation issue",
                    "booking issue",
                    "booking problem",
                    "wrong room",
                    "room not available",
                    "overbooked",
                    "overbooking",
                    "confirmation",
                    "booking confirmation",
                    "cancelled",
                    "canceled"
                ]
            ):

                labels.append(
                    "Booking / Reservation Issue"
                )


            # =================================================
            # Location
            # =================================================

            if any(
                keyword in text

                for keyword in [

                    "location",
                    "far from",
                    "too far",
                    "near",
                    "nearby",
                    "convenient location",
                    "inconvenient location",
                    "transport",
                    "train station",
                    "subway",
                    "mrt",
                    "bus",
                    "airport"
                ]
            ):

                labels.append(
                    "Location Issue"
                )


            # =================================================
            # Facilities / Amenities
            # =================================================

            if any(
                keyword in text

                for keyword in [

                    "facility",
                    "facilities",
                    "amenities",
                    "amenity",
                    "pool",
                    "swimming pool",
                    "gym",
                    "fitness",
                    "spa",
                    "sauna",
                    "parking",
                    "wifi",
                    "wi-fi",
                    "internet",
                    "elevator",
                    "lift"
                ]
            ):

                labels.append(
                    "Facilities / Amenities Issue"
                )


            # =================================================
            # Food / Breakfast
            # =================================================

            if any(
                keyword in text

                for keyword in [

                    "breakfast",
                    "food",
                    "restaurant",
                    "meal",
                    "dinner",
                    "lunch",
                    "buffet",
                    "coffee",
                    "food quality",
                    "bad food",
                    "poor food"
                ]
            ):

                labels.append(
                    "Food / Breakfast Issue"
                )


            # =================================================
            # Noise / Disturbance
            # =================================================

            if any(
                keyword in text

                for keyword in [

                    "noise",
                    "noisy",
                    "loud",
                    "too loud",
                    "disturbance",
                    "disturbed",
                    "music",
                    "party",
                    "construction",
                    "thin walls",
                    "neighbour",
                    "neighbor"
                ]
            ):

                labels.append(
                    "Noise / Disturbance Issue"
                )


            # =================================================
            # Bed / Comfort
            # =================================================

            if any(
                keyword in text

                for keyword in [

                    "bed",
                    "mattress",
                    "pillow",
                    "pillows",
                    "uncomfortable",
                    "uncomfortable bed",
                    "sleep",
                    "sleeping",
                    "room too small",
                    "small room",
                    "cramped",
                    "comfort"
                ]
            ):

                labels.append(
                    "Bed / Comfort Issue"
                )


            # =================================================
            # Value for Money
            # =================================================

            if any(
                keyword in text

                for keyword in [

                    "expensive",
                    "overpriced",
                    "price",
                    "value",
                    "value for money",
                    "not worth",
                    "worth the money",
                    "too expensive",
                    "cost",
                    "cheap"
                ]
            ):

                labels.append(
                    "Value for Money Issue"
                )


            # =================================================
            # DEFAULT
            # =================================================

            if not labels:

                labels.append(
                    "Other / Acceptable Stay"
                )


            return labels


        df["business_categories"] = (
            df["comment_clean"]
            .apply(
                extract_business
            )
        )


        # =====================================================
        # 14. SAMPLE COMMENTS
        # =====================================================

        df["sample_comments"] = (
            df["cluster"]
            .map(
                self.comment_cluster_samples
            )
        )


        # =====================================================
        # 15. COMPLETED
        # =====================================================

        print(
            "Feature Engineering - "
            "Step 5.6 APPLY Top Comment Clustering completed"
        )


        return df





