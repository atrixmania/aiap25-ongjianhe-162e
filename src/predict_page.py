# =========================
# src/predict_page.py
# =========================

import pandas as pd
import numpy as np
import joblib

from dash import html, dcc, Input, Output, State

from config import CONFIG


print("Predict Page Loading")


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

ml = joblib.load(
    CONFIG["service_model"]
)

CONFIG["model_type"] = ml.best_model_name

print(
    "[INFO] Prediction model:",
    ml.best_model_name
)


# =========================================================
# CATEGORICAL FEATURES
#
# IMPORTANT:
# These MUST remain human-readable strings.
#
# Example:
#     "city hotel"
#     "canceled"
#     "online-ta"
#
# NOT:
#     0
#     1
#     2
# =========================================================

CATEGORICAL_FEATURES = [

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


# =========================================================
# NUMERIC FEATURES
# =========================================================

NUMERIC_FEATURES = [

    "is_top_bad_client",
    "client_bad_rate",
    "is_canceled"

]


# =========================================================
# UI FEATURE LABELS
# =========================================================

FEATURE_LABELS = {

    "hotel":
        "Hotel Name",

    "country_group":
        "Customer Country Origin",

    "adr_group":
        "Average Daily Rate",

    "previous_cancellations_group":
        "No. of Prior Booking Cancellations",

    "days_in_waiting_list_group":
        "Waiting List Duration",

    "deposit_type":
        "Deposit Arrangement",

    "customer_type":
        "Type of Booking",

    "guest_group":
        "Type of Guest Group",

    "season_group":
        "Peak / Low Season",

    "meal":
        "Meal Package Booked",

    "market_segment":
        "Market Segment of Booking",

    "distribution_channel":
        "Booking Distribution Channel",

    "reservation_status":
        "Reservation Status",

    "is_canceled":
        "Is Reservation Cancelled"

}


# =========================================================
# FEATURES SHOWN ON PREDICTION PAGE
# =========================================================

FEATURES = [

    "hotel",
    "country_group",
    "adr_group",
    "reservation_status",
    "previous_cancellations_group",
    "days_in_waiting_list_group",
    "deposit_type",
    "customer_type",
    "guest_group",
    "season_group",
    "meal",
    "market_segment",
    "distribution_channel",
    "is_canceled"

]


# =========================================================
# HELPER
# =========================================================

def safe(
    value,
    default="unknown"
):

    if value is None:

        return default


    if isinstance(
        value,
        float
    ) and np.isnan(value):

        return default


    value = str(
        value
    ).strip()


    if value == "":

        return default


    if value.lower() in [
        "nan",
        "none",
        "null"
    ]:

        return default


    return value


# =========================================================
# CHECK WHETHER VALUE IS NUMERIC ENCODED CATEGORY
#
# Examples:
#
# "0" -> True
# "1" -> True
# "2" -> True
#
# "city hotel" -> False
# "canceled" -> False
# "online-ta" -> False
# =========================================================

def is_numeric_category(value):

    if value is None:

        return False


    try:

        value = str(
            value
        ).strip()


        if value == "":

            return False


        float(value)

        return True


    except (
        ValueError,
        TypeError
    ):

        return False


# =========================================================
# CLEAN CATEGORY VALUES
# =========================================================

def clean_category_values(
    values
):

    cleaned = []


    if values is None:

        return cleaned


    for value in values:

        if value is None:

            continue


        try:

            if pd.isna(value):

                continue

        except Exception:

            pass


        value = str(
            value
        ).strip()


        if value == "":

            continue


        if value.lower() in [
            "nan",
            "none",
            "null"
        ]:

            continue


        # =================================================
        # IMPORTANT:
        #
        # Never expose encoded numeric categories
        # such as 0, 1, 2 in categorical dropdowns.
        # =================================================

        if is_numeric_category(value):

            continue


        if value not in cleaned:

            cleaned.append(value)


    return cleaned


# =========================================================
# GET CATEGORIES FROM TRAINED MODEL
#
# We inspect the model for categories.
#
# If the model exposes encoded values such as:
#
#     ["0", "1"]
#
# those values are discarded.
#
# Human-readable categories are retained.
# =========================================================

def get_model_categories():

    categories = {}


    try:

        best_model = getattr(
            ml,
            "best_model",
            None
        )


        if best_model is None:

            print(
                "[WARNING] Best model is not available."
            )

            return categories


        # =================================================
        # RATING MODEL
        # =================================================

        if hasattr(
            best_model,
            "named_steps"
        ):

            prep = (
                best_model
                .named_steps
                .get("prep")
            )


            if prep is not None:

                transformers = getattr(
                    prep,
                    "transformers_",
                    []
                )


                for (
                    transformer_name,
                    transformer,
                    columns
                ) in transformers:

                    if transformer_name != "cat":

                        continue


                    encoder = transformer


                    # -------------------------------------------------
                    # Direct encoder
                    # -------------------------------------------------

                    if hasattr(
                        encoder,
                        "categories_"
                    ):

                        for (
                            column,
                            values
                        ) in zip(
                            columns,
                            encoder.categories_
                        ):

                            cleaned = (
                                clean_category_values(
                                    values
                                )
                            )


                            if cleaned:

                                categories[
                                    column
                                ] = cleaned


        # =================================================
        # STATUS MODEL
        # =================================================

        status_preprocess = getattr(
            ml,
            "status_preprocess",
            None
        )


        if status_preprocess is not None:

            transformers = getattr(
                status_preprocess,
                "transformers_",
                []
            )


            for (
                transformer_name,
                transformer,
                columns
            ) in transformers:

                if transformer_name != "cat":

                    continue


                encoder = transformer


                if hasattr(
                    encoder,
                    "categories_"
                ):

                    for (
                        column,
                        values
                    ) in zip(
                        columns,
                        encoder.categories_
                    ):

                        cleaned = (
                            clean_category_values(
                                values
                            )
                        )


                        if (
                            cleaned
                            and column not in categories
                        ):

                            categories[
                                column
                            ] = cleaned


    except Exception as e:

        print(
            "[WARNING] Unable to read model categories:",
            e
        )


    return categories


# =========================================================
# MODEL CATEGORIES
# =========================================================

MODEL_CATEGORIES = (
    get_model_categories()
)


# =========================================================
# FALLBACK CATEGORIES
#
# IMPORTANT:
# These are HUMAN-READABLE values.
#
# They are only used when the trained model does not
# provide usable human-readable categories.
# =========================================================

FALLBACK_CATEGORIES = {

    "hotel": [

        "city hotel",
        "airport hotel"

    ],


    "meal": [

        "BB (Bed & Breakfast)",
        "HB (Half Board)",
        "FB (Full Board)",
        "SC (Self Catering)",
        "undefined"

    ],


    "market_segment": [

        "offline-ta/to",
        "online-ta",
        "groups",
        "direct",
        "corporate",
        "complementary",
        "aviation",
        "undefined"

    ],


    "distribution_channel": [

        "direct",
        "corporate",
        "gds",
        "undefined",
        "ta/to"

    ],


    "deposit_type": [

        "non-refund",
        "no-deposit",
        "refundable"

    ],


    "customer_type": [

        "transient",
        "transient-party",
        "contract",
        "group"

    ],


    "reservation_status": [

        "canceled",
        "no-show",
        "check-out"

    ],


    "season_group": [

        "unknown"

    ],


    "country_group": [

        "local",
        "international",
        "unknown"

    ],


    "adr_group": [

        "Low value (≤100)",
        "Medium value (101-200)",
        "High value (>200)",
        "unknown"

    ],


    "previous_cancellations_group": [

        "unknown"

    ],


    "days_in_waiting_list_group": [

        "unknown"

    ],


    "guest_group": [

        "unknown"

    ],


    "client_segment": [

        "low-risk",
        "medium-risk",
        "high-risk"

    ]

}


# =========================================================
# GET DROPDOWN CATEGORIES
# =========================================================

def get_categories(
    column
):

    # =====================================================
    # First try model categories
    # =====================================================

    values = MODEL_CATEGORIES.get(
        column,
        []
    )


    values = clean_category_values(
        values
    )


    # =====================================================
    # If model only had numeric encoded categories,
    # use human-readable fallback.
    # =====================================================

    if not values:

        values = FALLBACK_CATEGORIES.get(
            column,
            []
        )


    values = clean_category_values(
        values
    )


    return values


# =========================================================
# BUILD CATEGORICAL OPTIONS
#
# label = what user sees
# value = exact STRING sent to model.py
# =========================================================

def build_categorical_options(
    column
):

    values = get_categories(
        column
    )


    options = []


    for value in values:

        # -------------------------------------------------
        # Make UI label nicer
        # -------------------------------------------------

        label = (
            str(value)
            .replace(
                "_",
                " "
            )
            .title()
        )


        options.append({

            "label":
                label,

            # IMPORTANT:
            # Keep actual category string.
            "value":
                str(value)

        })


    return options


# =========================================================
# NUMERIC OPTIONS
# =========================================================

def build_numeric_options(
    column
):

    if column == "is_canceled":

        return [

            {
                "label":
                    "No",

                "value":
                    0

            },

            {
                "label":
                    "Yes",

                "value":
                    1

            }

        ]


    return []


# =========================================================
# DEBUG MODEL CATEGORIES
# =========================================================

print()
print("==============================")
print("MODEL CATEGORICAL CATEGORIES")
print("==============================")


for column in CATEGORICAL_FEATURES:

    print(
        f"{column}:",
        get_categories(column)
    )


print("==============================")
print("END MODEL CATEGORIES")
print("==============================")
print()


# =========================================================
# CREATE PAGE LAYOUT
# =========================================================

def create_layout(
    df
):

    TOP_CLIENTS = df.attrs.get(
        "top_10_bad_clients",
        []
    )


    # =====================================================
    # TOP BAD CLIENT OPTIONS
    # =====================================================

    top_client_options = [

        {
            "label":
                str(value),

            "value":
                str(value)

        }

        for value in TOP_CLIENTS

        if value is not None

    ]


    # =====================================================
    # FEATURE CONTROLS
    # =====================================================

    feature_controls = []


    for column in FEATURES:

        label = FEATURE_LABELS.get(
            column,
            column
        )


        if column in CATEGORICAL_FEATURES:

            options = (
                build_categorical_options(
                    column
                )
            )

        else:

            options = (
                build_numeric_options(
                    column
                )
            )


        feature_controls.append(

            html.Div(

                [

                    html.Label(
                        label
                    ),

                    dcc.Dropdown(

                        id=column,

                        options=options,

                        value=None,

                        clearable=True,

                        placeholder=(
                            f"Select {label}"
                        )

                    )

                ],

                style={
                    "marginBottom":
                        "10px"
                }

            )

        )


    return html.Div(

        [

            # =================================================
            # TITLE
            # =================================================

            html.H2(

                "Service Level Prediction Dashboard",

                style={
                    "textAlign":
                        "center"
                }

            ),


            # =================================================
            # BACK BUTTON
            # =================================================

            html.Div(

                [

                    dcc.Link(

                        html.Button(
                            "⬅ Back to Main"
                        ),

                        href="/"

                    )

                ],

                style={
                    "marginBottom":
                        "15px"
                }

            ),


            # =================================================
            # MODEL SELECTOR
            # =================================================

            html.Div(

                [

                    html.Label(
                        "Model Type"
                    ),

                    dcc.Dropdown(

                        id="model_choice",

                        options=[

                            {
                                "label":
                                    "Logistic Regression",

                                "value":
                                    "Logistic Regression"

                            },

                            {
                                "label":
                                    "Linear SVC",

                                "value":
                                    "Linear SVC"

                            },

                            {
                                "label":
                                    "LightGBM",

                                "value":
                                    "LightGBM"

                            }

                        ],

                        value=ml.best_model_name,

                        clearable=False

                    )

                ],

                style={
                    "marginBottom":
                        "15px"
                }

            ),


            # =================================================
            # TOP BAD CLIENT
            # =================================================

            html.Div(

                [

                    html.Label(
                        "⭐ Top 10 Clients with frequent bad review"
                    ),

                    dcc.Dropdown(

                        id="top_bad_client",

                        options=top_client_options,

                        value=None,

                        clearable=True,

                        placeholder=
                            "Select client"

                    ),

                    html.Div(

                        id=
                            "client_bad_rate_display",

                        style={

                            "marginTop":
                                "8px",

                            "fontWeight":
                                "bold"

                        }

                    )

                ],

                style={
                    "marginBottom":
                        "15px"
                }

            ),


            # =================================================
            # FEATURES
            # =================================================

            html.Div(
                feature_controls
            ),


            # =================================================
            # PREDICT BUTTON
            # =================================================

            html.Button(

                "Predict",

                id="predict_btn",

                n_clicks=0,

                style={
                    "marginTop":
                        "15px"
                }

            ),


            # =================================================
            # OUTPUT
            # =================================================

            html.Div(

                id="output",

                style={
                    "marginTop":
                        "20px"
                }

            )

        ]

    )


# =========================================================
# CLIENT BAD RATE
# =========================================================

def calculate_client_bad_rate(
    df,
    client
):

    if client is None:

        return 0.0


    client = str(
        client
    ).strip().lower()


    if client == "":

        return 0.0


    # =====================================================
    # FIND CLIENT COLUMN
    # =====================================================

    possible_columns = [

        "client_email",
        "email",
        "customer_email"

    ]


    client_column = None


    for column in possible_columns:

        if column in df.columns:

            client_column = column

            break


    if client_column is None:

        return 0.0


    client_mask = (

        df[client_column]
        .astype(str)
        .str.strip()
        .str.lower()
        == client

    )


    client_df = df[
        client_mask
    ].copy()


    if len(client_df) == 0:

        return 0.0


    # =====================================================
    # FIND RATING COLUMN
    # =====================================================

    if "final_rating" not in client_df.columns:

        return 0.0


    ratings = pd.to_numeric(

        client_df[
            "final_rating"
        ],

        errors="coerce"

    )


    ratings = ratings.dropna()


    if len(ratings) == 0:

        return 0.0


    # =====================================================
    # BAD RATING
    #
    # 0, 1, 2 = BAD
    # =====================================================

    bad_count = int(
        (ratings <= 2).sum()
    )


    total_count = len(
        ratings
    )


    if total_count == 0:

        return 0.0


    return float(
        bad_count /
        total_count
    )


# =========================================================
# CLIENT SEGMENT
# =========================================================

def calculate_client_segment(
    bad_rate
):

    bad_rate = float(
        bad_rate
    )


    if bad_rate >= 0.50:

        return "high-risk"


    elif bad_rate >= 0.20:

        return "medium-risk"


    return "low-risk"


# =========================================================
# NUMERIC SAFETY
# =========================================================

def safe_numeric(
    value,
    default=0.0
):

    try:

        if value is None:

            return float(
                default
            )


        if isinstance(
            value,
            str
        ):

            value = value.strip()

            if value == "":

                return float(
                    default
                )


        value = float(
            value
        )


        if np.isnan(value):

            return float(
                default
            )


        return value


    except Exception:

        return float(
            default
        )


# =========================================================
# REGISTER CALLBACKS
# =========================================================

def register_callbacks(
    app,
    df
):

    TOP_CLIENTS = df.attrs.get(
        "top_10_bad_clients",
        []
    )


    # =====================================================
    # CLIENT BAD RATE CALLBACK
    # =====================================================

    @app.callback(

        Output(
            "client_bad_rate_display",
            "children"
        ),

        Input(
            "top_bad_client",
            "value"
        )

    )

    def update_client_bad_rate(
        top_bad_client
    ):

        if not top_bad_client:

            return ""


        bad_rate = (
            calculate_client_bad_rate(
                df,
                top_bad_client
            )
        )


        segment = (
            calculate_client_segment(
                bad_rate
            )
        )


        return (

            f"Client Bad Rate: "
            f"{bad_rate * 100:.1f}% "
            f"| Segment: {segment}"

        )


    # =====================================================
    # PREDICTION CALLBACK
    # =====================================================

    @app.callback(

        Output(
            "output",
            "children"
        ),

        Input(
            "predict_btn",
            "n_clicks"
        ),

        State(
            "model_choice",
            "value"
        ),

        State(
            "top_bad_client",
            "value"
        ),

        *[

            State(
                column,
                "value"
            )

            for column in FEATURES

        ]

    )

    def predict(

        n,

        model_choice,

        top_bad_client,

        hotel,

        country_group,

        adr_group,

        reservation_status,

        previous_cancellations_group,

        days_in_waiting_list_group,

        deposit_type,

        customer_type,

        guest_group,

        season_group,

        meal,

        market_segment,

        distribution_channel,

        is_canceled

    ):

        # =================================================
        # NO CLICK
        # =================================================

        if not n:

            return ""


        # =================================================
        # SWITCH MODEL
        # =================================================

        if model_choice:

            ml.set_model(
                model_choice
            )


        # =================================================
        # CLIENT BAD RATE
        # =================================================

        client_bad_rate = (
            calculate_client_bad_rate(
                df,
                top_bad_client
            )
        )


        # =================================================
        # CLIENT SEGMENT
        # =================================================

        client_segment = (
            calculate_client_segment(
                client_bad_rate
            )
        )


        # =================================================
        # TOP BAD CLIENT
        # =================================================

        top_client_strings = {

            str(value)
            .strip()
            .lower()

            for value in TOP_CLIENTS

            if value is not None

        }


        selected_client_key = (

            str(
                top_bad_client
            )
            .strip()
            .lower()

            if top_bad_client is not None

            else ""

        )


        if (
            selected_client_key
            in top_client_strings
        ):

            is_top_bad_client = 1.0

        else:

            is_top_bad_client = 0.0


        # =================================================
        # BUILD PREDICTION INPUT
        #
        # CATEGORICAL VALUES REMAIN STRINGS.
        # =================================================

        x = {

            "comment":
                "",

            "client_email":
                safe(
                    top_bad_client
                ),

            "client_segment":
                safe(
                    client_segment
                ),

            "is_top_bad_client":
                is_top_bad_client,

            "client_bad_rate":
                client_bad_rate,


            # =================================================
            # CATEGORICAL STRING FEATURES
            # =================================================

            "hotel":
                safe(hotel),

            "country_group":
                safe(country_group),

            "adr_group":
                safe(adr_group),

            "previous_cancellations_group":
                safe(
                    previous_cancellations_group
                ),

            "days_in_waiting_list_group":
                safe(
                    days_in_waiting_list_group
                ),

            "deposit_type":
                safe(
                    deposit_type
                ),

            "customer_type":
                safe(
                    customer_type
                ),

            "guest_group":
                safe(
                    guest_group
                ),

            "season_group":
                safe(
                    season_group
                ),

            "meal":
                safe(
                    meal
                ),

            "market_segment":
                safe(
                    market_segment
                ),

            "distribution_channel":
                safe(
                    distribution_channel
                ),

            "reservation_status":
                safe(
                    reservation_status
                ),


            # =================================================
            # NUMERIC FEATURE
            # =================================================

            "is_canceled":
                safe_numeric(
                    is_canceled,
                    0.0
                )

        }


        # =================================================
        # FORCE CATEGORICAL VALUES TO STRINGS
        #
        # NEVER convert these to 0 / 1 / 2.
        # =================================================

        for column in CATEGORICAL_FEATURES:

            value = x.get(
                column
            )


            if value is None:

                x[column] = "unknown"

            else:

                x[column] = str(
                    value
                ).strip()


        # =================================================
        # FORCE NUMERIC VALUES
        # =================================================

        x["is_top_bad_client"] = (
            safe_numeric(
                x["is_top_bad_client"],
                0.0
            )
        )


        x["client_bad_rate"] = (

            safe_numeric(
                x["client_bad_rate"],
                0.0
            )

        )


        x["client_bad_rate"] = np.clip(

            x["client_bad_rate"],

            0.0,

            1.0

        )


        x["is_canceled"] = (

            safe_numeric(
                x["is_canceled"],
                0.0
            )

        )


        x["is_canceled"] = np.clip(

            x["is_canceled"],

            0.0,

            1.0

        )


        # =================================================
        # RESERVATION STATUS NORMALISATION
        #
        # KEEP AS STRING.
        # =================================================

        status = (

            str(
                x["reservation_status"]
            )
            .strip()
            .lower()

        )


        if status in [

            "cancelled",
            "canceled"

        ]:

            status = "canceled"


        elif status in [

            "checkout",
            "check out",
            "check-out"

        ]:

            status = "check-out"


        elif status in [

            "noshow",
            "no show",
            "no-show"

        ]:

            status = "no-show"


        elif status in [

            "",
            "nan",
            "none",
            "null",
            "unknown"

        ]:

            status = "unknown"


        x["reservation_status"] = status


        # =================================================
        # IS CANCELED CONSISTENCY
        # =================================================

        if status == "canceled":

            x["is_canceled"] = 1.0


        elif status in [

            "no-show",
            "check-out"

        ]:

            x["is_canceled"] = 0.0


        # =================================================
        # DEBUG INPUT
        # =================================================

        print()
        print("==============================")
        print("PREDICTION INPUT")
        print("==============================")


        for key, value in x.items():

            print(

                f"{key}: "
                f"{value!r} "
                f"(type={type(value).__name__})"

            )


        print()

        print(
            "RESERVATION STATUS SELECTED:",
            x["reservation_status"]
        )

        print(
            "RESERVATION STATUS TYPE:",
            type(
                x["reservation_status"]
            ).__name__
        )

        print(
            "IS CANCELED VALUE:",
            x["is_canceled"]
        )


        # =================================================
        # PREDICT
        # =================================================

        (

            pred,

            prob_canceled,

            prob_check_out,

            prob_no_show,

            sla_risk,

            risk,

            comment

        ) = ml.predict(
            x
        )


        # =================================================
        # RATING LEVEL
        # =================================================

        rating_levels = {

            0:
                "Severe dissatisfaction",

            1:
                "Poor",

            2:
                "Needs Improvement",

            3:
                "Average",

            4:
                "Positive",

            5:
                "Very Good",

            6:
                "Excellent"

        }


        rating_level = (
            rating_levels.get(
                int(pred),
                "Unknown"
            )
        )


        # =================================================
        # OUTPUT
        # =================================================

        return html.Div(

            [

                html.H4(
                    f"Customer Rating: "
                    f"{pred} / 6"
                ),

                html.H4(
                    f"Rating Level: "
                    f"{rating_level}"
                ),

                html.H4(
                    f"Customer Risk Score: "
                    f"{risk}"
                ),

                html.H4(
                    f"Client Bad Rate: "
                    f"{client_bad_rate * 100:.1f}%"
                ),

                html.H4(
                    f"Is Canceled: "
                    f"{x['is_canceled']}"
                ),

                html.H4(
                    f"Cancellation: "
                    f"{prob_canceled}%"
                ),

                html.H4(
                    f"Check Out: "
                    f"{prob_check_out}%"
                ),

                html.H4(
                    f"No Show: "
                    f"{prob_no_show}%"
                ),

                html.H4(
                    f"Service Level Risk: "
                    f"{sla_risk}%"
                ),

                html.Hr(),

                html.P(
                    comment
                )

            ]

        )


# =========================================================
# END OF predict_page.py
# =========================================================












