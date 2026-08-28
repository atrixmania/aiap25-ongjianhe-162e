# =========================
# src/predict_page.py
# =========================

import pandas as pd
import numpy as np
import joblib

from model import ServiceModel
from data_loader import load_data, load_prediction_data
from config import CONFIG

from dash import html, dcc, Input, Output, State

print("Predict Page Loading")

# =========================================================
# LOAD TRAINED MODEL
# =========================================================

ml = joblib.load(
    CONFIG["service_model"]
)

CONFIG["model_type"] = ml.best_model_name

# =========================================================
# RATING DEFINITIONS
# =========================================================

RATING_LABELS = {

    0: "Very Poor",
    1: "Poor",
    2: "Below Average",
    3: "Average",
    4: "Good",
    5: "Very Good",
    6: "Excellent"

}

# =========================================================
# RATING RISK
# =========================================================

RATING_RISK = {

    0: 100,
    1: 90,
    2: 75,
    3: 60,
    4: 40,
    5: 20,
    6: 10

}

# =========================================================
# MAPPINGS
# =========================================================

MAPPINGS = {

    "hotel": {
        0: "city hotel",
        1: "airport hotel"
    },

    "meal": {
        0: "BB (Bed & Breakfast)",
        1: "HB (Half Board)",
        2: "FB (Full Board)",
        3: "SC (Self Catering)",
        4: "undefined"
    },

    "market_segment": {
        0: "offline-ta/to",
        1: "online-ta",
        2: "groups",
        3: "direct",
        4: "corporate",
        5: "complementary",
        6: "aviation",
        7: "undefined"
    },

    "distribution_channel": {
        0: "direct",
        1: "corporate",
        2: "gds",
        3: "undefined",
        4: "ta/to"
    },

    "deposit_type": {
        0: "non-refund",
        1: "no-deposit",
        2: "refundable"
    },

    "customer_type": {
        0: "transient",
        1: "transient-party",
        2: "contract",
        3: "group"
    },

    "reservation_status": {
        0: "canceled",
        1: "no-show",
        2: "check-out"
    }

}

# =========================================================
# FEATURE LABELS
# =========================================================

FEATURE_LABELS = {

    "hotel": "Hotel Name",

    "meal": "Meal Package Booked",

    "market_segment": "Market Segment of Booking",

    "distribution_channel":
        "Booking Distribution Channel",

    "deposit_type":
        "Deposit Arrangement",

    "customer_type":
        "Type of Booking",

    "season_group":
        "Peak / Low Season",

    "country_group":
        "Customer Country Origin",

    "adr_group":
        "Average Daily Rate",

    "previous_cancellations_group":
        "No of Prior Booking Cancellation",

    "days_in_waiting_list_group":
        "Waiting list duration",

    "guest_group":
        "Type of Guest Group",

    "reservation_status":
        "Reservation Status",

    "is_canceled":
        "Is Reservation Cancel"

}

# =========================================================
# FEATURES
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
# HELPER FUNCTIONS
# =========================================================

def build_options(mapping):

    options = []

    for value in mapping.values():

        if value is None:
            continue

        value = str(value)

        options.append({
            "label": value.title(),
            "value": value
        })

    return options


# =========================================================
# SAFE CATEGORICAL VALUE
# =========================================================

def safe(v):

    if v is None:
        return "unknown"

    if isinstance(v, str):

        value = v.strip()

        if value == "":
            return "unknown"

        if value.lower() in [
            "nan",
            "none",
            "null",
            "missing"
        ]:
            return "unknown"

        return value

    return v


# =========================================================
# SAFE NUMERIC VALUE
# =========================================================

def safe_numeric(v, default=0.0):

    if v is None:
        return default

    if isinstance(v, str):

        value = v.strip().lower()

        if value in [
            "",
            "nan",
            "none",
            "null",
            "missing",
            "unknown"
        ]:
            return default

    try:

        number = float(v)

        if np.isnan(number):
            return default

        return number

    except Exception:

        return default


# =========================================================
# CLEAN VALUES
# =========================================================

def clean_values(series):

    if series is None:
        return []

    values = []

    for value in series:

        if value is None:
            continue

        if isinstance(value, float) and np.isnan(value):
            continue

        value = str(value).strip()

        if value.lower() in [
            "missing",
            "unknown",
            "nan",
            "none",
            "null",
            ""
        ]:
            continue

        values.append(value)

    return sorted(
        list(set(values))
    )


# =========================================================
# FIND CLIENT COLUMN
# =========================================================

def get_client_column(df):

    possible_columns = [

        "client_email",
        "email",
        "customer_email",
        "client"

    ]

    for col in possible_columns:

        if col in df.columns:
            return col

    return None


# =========================================================
# CALCULATE CLIENT BAD RATE
# =========================================================

def calculate_client_bad_rate(
    df,
    client_value
):

    if df is None:
        return 0.0

    if client_value is None:
        return 0.0

    client_value = str(
        client_value
    ).strip()

    if client_value == "":
        return 0.0

    client_column = get_client_column(
        df
    )

    if client_column is None:

        print(
            "[WARNING] Client column not found. "
            "client_bad_rate = 0.0"
        )

        return 0.0

    if "final_rating" not in df.columns:

        print(
            "[WARNING] final_rating column not found. "
            "client_bad_rate = 0.0"
        )

        return 0.0

    work_df = df.copy()

    work_df["_client_key"] = (

        work_df[client_column]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()

    )

    selected_key = (
        client_value
        .lower()
        .strip()
    )

    client_df = work_df[
        work_df["_client_key"] == selected_key
    ].copy()

    if len(client_df) == 0:

        print(
            f"[INFO] No historical records found "
            f"for client: {client_value}"
        )

        return 0.0

    client_df["_rating_numeric"] = pd.to_numeric(

        client_df["final_rating"],

        errors="coerce"

    )

    client_df = client_df[
        client_df["_rating_numeric"].notna()
    ]

    if len(client_df) == 0:
        return 0.0

    bad_count = (

        client_df["_rating_numeric"]
        .isin([0, 1, 2])
        .sum()

    )

    total_count = len(
        client_df
    )

    if total_count == 0:
        return 0.0

    bad_rate = (
        bad_count /
        total_count
    )

    bad_rate = float(
        np.clip(
            bad_rate,
            0.0,
            1.0
        )
    )

    print()
    print(
        "=============================="
    )
    print(
        "CLIENT BAD RATE"
    )
    print(
        "=============================="
    )
    print(
        f"Client        : {client_value}"
    )
    print(
        f"Total records : {total_count}"
    )
    print(
        f"Bad records   : {bad_count}"
    )
    print(
        f"Bad rate      : {bad_rate:.4f}"
    )
    print(
        f"Bad rate %    : {bad_rate * 100:.1f}%"
    )

    return bad_rate


# =========================================================
# CALCULATE CLIENT SEGMENT
# =========================================================

def calculate_client_segment(
    bad_rate
):

    bad_rate = safe_numeric(
        bad_rate,
        0.0
    )

    if bad_rate >= 0.70:

        return "high-risk"

    elif bad_rate >= 0.40:

        return "bad"

    elif bad_rate >= 0.20:

        return "moderate"

    else:

        return "low-risk"


# =========================================================
# CREATE PAGE LAYOUT
# =========================================================

def create_layout(df):

    TOP_CLIENTS = df.attrs.get(
        "top_10_bad_clients",
        []
    )

    return html.Div([

        html.H2(
            "Service Level Prediction Dashboard",
            style={
                "textAlign": "center"
            }
        ),

        html.Div([

            dcc.Link(
                html.Button(
                    "⬅ Back to Main"
                ),
                href="/"
            )

        ], style={
            "marginBottom": "15px"
        }),

        html.Div([

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

                value=ml.best_model_name

            )

        ], style={
            "marginBottom": "15px"
        }),

        html.Div([

            html.Label(
                "⭐ Top 10 Clients with frequent bad review"
            ),

            dcc.Dropdown(

                id="top_bad_client",

                options=[

                    {
                        "label": str(v),
                        "value": str(v)
                    }

                    for v in TOP_CLIENTS

                ],

                value=None,

                placeholder=
                    "Select top bad client"

            )

        ], style={
            "marginBottom": "15px"
        }),

        html.Div([

            html.Div([

                html.Label(
                    FEATURE_LABELS.get(
                        col,
                        col
                    )
                ),

                dcc.Dropdown(

                    id=col,

                    options=(

                        build_options(
                            MAPPINGS[col]
                        )

                        if col in MAPPINGS

                        else [

                            {
                                "label": v,
                                "value": v
                            }

                            for v in clean_values(
                                df[col]
                            )

                        ]

                    ),

                    value=None,

                    placeholder=
                        "Select " +
                        FEATURE_LABELS.get(
                            col,
                            col
                        )

                )

            ], style={
                "marginBottom": "10px"
            })

            for col in FEATURES

        ]),

        html.Button(
            "Predict",
            id="predict_btn"
        ),

        html.H3(
            id="output"
        )

    ])


# =========================================================
# REGISTER CALLBACKS
# =========================================================

def register_callbacks(app, df):

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

        State(
            "hotel",
            "value"
        ),

        State(
            "country_group",
            "value"
        ),

        State(
            "adr_group",
            "value"
        ),

        State(
            "previous_cancellations_group",
            "value"
        ),

        State(
            "days_in_waiting_list_group",
            "value"
        ),

        State(
            "deposit_type",
            "value"
        ),

        State(
            "customer_type",
            "value"
        ),

        State(
            "guest_group",
            "value"
        ),

        State(
            "season_group",
            "value"
        ),

        State(
            "meal",
            "value"
        ),

        State(
            "market_segment",
            "value"
        ),

        State(
            "distribution_channel",
            "value"
        ),

        State(
            "reservation_status",
            "value"
        ),

        State(
            "is_canceled",
            "value"
        )

    )
    def predict(

        n,
        model_choice,
        top_bad_client,
        hotel,
        country_group,
        adr_group,
        previous_cancellations_group,
        days_in_waiting_list_group,
        deposit_type,
        customer_type,
        guest_group,
        season_group,
        meal,
        market_segment,
        distribution_channel,
        reservation_status,
        is_canceled

    ):

        if not n:
            return ""

        try:

            # =================================================
            # SELECT FINAL RATING MODEL
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

            TOP_CLIENTS = df.attrs.get(
                "top_10_bad_clients",
                []
            )

            top_client_strings = {

                str(v)
                .strip()
                .lower()

                for v in TOP_CLIENTS

                if v is not None

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

            if selected_client_key in top_client_strings:

                is_top_bad_client = 1.0

            else:

                is_top_bad_client = 0.0

            # =================================================
            # BUILD MODEL INPUT
            #
            # IMPORTANT:
            #
            # The UI sends ALL values to model.py.
            #
            # model.py decides which values are used for:
            #
            # 1. reservation status prediction
            # 2. final rating prediction
            # 3. SLA risk
            # 4. customer risk
            # 5. comment generation
            # =================================================

            x = {

                "comment":
                    "",

                "client_email":
                    safe(
                        top_bad_client
                    ),

                "client_segment":
                    client_segment,

                "is_top_bad_client":
                    is_top_bad_client,

                "client_bad_rate":
                    client_bad_rate,

                "hotel":
                    safe(
                        hotel
                    ),

                "country_group":
                    safe(
                        country_group
                    ),

                "adr_group":
                    safe(
                        adr_group
                    ),

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

                "is_canceled":
                    safe_numeric(
                        is_canceled,
                        0.0
                    )

            }

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

            x["is_canceled"] = (
                safe_numeric(
                    x["is_canceled"],
                    0.0
                )
            )

            # =================================================
            # DEBUG INPUT
            # =================================================

            print()
            print(
                "=============================="
            )
            print(
                "PREDICTION INPUT"
            )
            print(
                "=============================="
            )

            for key, value in x.items():

                print(
                    f"{key}: {value!r} "
                    f"(type={type(value).__name__})"
                )

            # =================================================
            # MODEL PREDICTION
            #
            # model.py is responsible for deciding where
            # each input is used.
            #
            # Expected return:
            #
            # pred
            # prob_canceled
            # prob_check_out
            # prob_no_show
            # sla_risk
            # risk
            # comment
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
            # VALIDATE FINAL RATING
            # =================================================

            pred = int(
                pred
            )

            if pred not in [
                0,
                1,
                2,
                3,
                4,
                5,
                6
            ]:

                raise ValueError(
                    f"Invalid rating returned by model: "
                    f"{pred}. Expected 0-6."
                )

            rating_label = (
                RATING_LABELS.get(
                    pred,
                    "Unknown"
                )
            )

            # =================================================
            # CONVERT MODEL RESULTS
            # =================================================

            risk = float(
                risk
            )

            sla_risk = float(
                sla_risk
            )

            prob_canceled = float(
                prob_canceled
            )

            prob_check_out = float(
                prob_check_out
            )

            prob_no_show = float(
                prob_no_show
            )

            # =================================================
            # SAFETY CLIP
            #
            # Probability values must remain 0-100.
            # =================================================

            prob_canceled = float(
                np.clip(
                    prob_canceled,
                    0.0,
                    100.0
                )
            )

            prob_check_out = float(
                np.clip(
                    prob_check_out,
                    0.0,
                    100.0
                )
            )

            prob_no_show = float(
                np.clip(
                    prob_no_show,
                    0.0,
                    100.0
                )
            )

            sla_risk = float(
                np.clip(
                    sla_risk,
                    0.0,
                    100.0
                )
            )

            # =================================================
            # DEBUG RESULT
            # =================================================

            print()
            print(
                "=============================="
            )
            print(
                "PREDICTION RESULT"
            )
            print(
                "=============================="
            )

            print(
                f"Customer Rating       : {pred} / 6"
            )

            print(
                f"Rating Level          : {rating_label}"
            )

            print(
                f"Customer Risk Score   : {risk:.1f}"
            )

            print(
                f"Client Bad Rate       : "
                f"{client_bad_rate * 100:.1f}%"
            )

            print(
                f"Is Canceled           : "
                f"{x['is_canceled']}"
            )

            print(
                f"Cancellation         : "
                f"{prob_canceled:.1f}%"
            )

            print(
                f"Check Out             : "
                f"{prob_check_out:.1f}%"
            )

            print(
                f"No Show               : "
                f"{prob_no_show:.1f}%"
            )

            print(
                f"Service Level Risk    : "
                f"{sla_risk:.1f}%"
            )

            # =================================================
            # OUTPUT
            # =================================================

            return html.Div([

                html.H4(
                    f"Customer Rating: "
                    f"{pred} / 6"
                ),

                html.H4(
                    f"Rating Level: "
                    f"{rating_label}"
                ),

                html.H4(
                    f"Customer Risk Score: "
                    f"{risk:.1f}"
                ),

                html.H4(
                    f"Client Bad Rate: "
                    f"{client_bad_rate * 100:.1f}%"
                ),

                html.H4(
                    f"Cancellation Probability: "
                    f"{prob_canceled:.1f}%"
                ),

                html.H4(
                    f"Check Out Probability: "
                    f"{prob_check_out:.1f}%"
                ),

                html.H4(
                    f"No Show Probability: "
                    f"{prob_no_show:.1f}%"
                ),

                html.H4(
                    f"Service Level Risk: "
                    f"{sla_risk:.1f}%"
                ),

                html.P(
                    comment
                )

            ])

        # =====================================================
        # PREDICTION ERROR
        # =====================================================

        except Exception as e:

            import traceback

            print()
            print(
                "=============================="
            )
            print(
                "PREDICTION ERROR"
            )
            print(
                "=============================="
            )

            traceback.print_exc()

            return html.Div([

                html.H4(
                    "Prediction Error",
                    style={
                        "color": "red"
                    }
                ),

                html.P(
                    str(e),
                    style={
                        "color": "red"
                    }
                )

            ])


# =========================================================
# END OF predict_page.py
# =========================================================









