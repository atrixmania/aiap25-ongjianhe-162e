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
# MAPPINGS
# =========================================================

MAPPINGS = {

    "hotel": {0: 'city hotel', 1: 'airport hotel'},
    "meal": {0: 'BB (Bed & Breakfast)', 1: 'HB (Half Board)', 2: 'FB (Full Board)', 3: 'SC (Self Catering)', 4: 'undefined', 5: np.nan},
    "market_segment": {0: 'offline-ta/to', 1: 'online-ta', 2: 'groups', 3: 'direct', 4: 'corporate', 
                        5: 'complementary', 6: 'aviation', 7: 'undefined', 8: np.nan},
    "distribution_channel": {0: 'direct', 1: 'corporate', 2: 'gds', 3: 'undefined', 4: 'ta/to', 5: np.nan},
    "deposit_type": {0: 'non-refund', 1: 'no-deposit', 2: 'refundable'},
    "customer_type": {0: 'transient', 1: 'transient-party', 2: 'contract', 3: 'group', 4: np.nan},
    "reservation_status": {0: 'canceled', 1: 'no-show', 2: 'check-out'}
}


# =========================================================
# FEATURES
# =========================================================

FEATURE_LABELS = {

    "hotel":
        "Hotel Name",

    "meal":
        "Meal Package Booked",

    "market_segment":
        "Market Segment of Booking",

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

    for v in mapping.values():

        # Handle missing / NaN values
        if pd.isna(v):

            options.append({
                "label": "Missing",
                "value": "missing"
            })

        else:

            v = str(v)

            options.append({
                "label": v.title(),
                "value": v
            })

    return options


def safe(v):

    return (
        v
        if v not in [None, "", "nan"]
        else "missing"
    )


def clean_values(series):

    return sorted([

        v

        for v in series.dropna()
        .astype(str)
        .unique()

        if v.lower()
        not in [
            "missing",
            "unknown",
            "nan",
            "none",
            ""
        ]

    ])


# =========================================================
# CREATE PAGE LAYOUT
# =========================================================

def create_layout(df):

    # =====================================================
    # TOP BAD CLIENTS / DRIVERS
    # =====================================================

    TOP_CLIENTS = df.attrs.get(
        "top_10_bad_clients",
        []
    )



    # =====================================================
    # PAGE LAYOUT
    # =====================================================

    return html.Div([

        # =================================================
        # TITLE
        # =================================================

        html.H2(
            "Service Level Prediction Dashboard",
            style={
                "textAlign": "center"
            }
        ),


        # =================================================
        # BACK BUTTON
        # =================================================

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


        # =================================================
        # MODEL SELECTOR
        # =================================================

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
                    },

                ],

                value=ml.best_model_name

            )

        ], style={
            "marginBottom": "15px"
        }),


        # =================================================
        # TOP BAD CLIENT
        # =================================================

        html.Div([

            html.Label(
                "⭐ Top 10 Clients with frequent bad review"
            ),

            dcc.Dropdown(

                id="top_bad_client",

                options=[

                    {
                        "label": str(v),
                        "value": v
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




        # =================================================
        # FEATURE DROPDOWNS
        # =================================================

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

                    value=None

                )

            ], style={
                "marginBottom": "10px"
            })

            for col in FEATURES

        ]),


        # =================================================
        # PREDICT BUTTON
        # =================================================

        html.Button(
            "Predict",
            id="predict_btn"
        ),


        # =================================================
        # OUTPUT
        # =================================================

        html.H3(
            id="output"
        )

    ])


# =========================================================
# REGISTER CALLBACKS
# =========================================================

def register_callbacks(app, df):

    # =====================================================
    # TOP BAD CLIENTS / DRIVERS
    # =====================================================

    TOP_CLIENTS = df.attrs.get(
        "top_10_bad_clients",
        []
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
        # BUILD INPUT
        # =================================================

        x = {

            "comment": "",

            "client_email":
                safe(
                    top_bad_client
                ),


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
                safe(
                    is_canceled
                )
            

            
        }


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
        )= ml.predict(x)


        # =================================================
        # OUTPUT
        # =================================================

        return html.Div([

            html.H4(
                f"Customer Rating: {pred}"
            ),

            html.H4(
                f"Customer Risk Score: {risk}"
            ),


            html.H4(
                f"Cancellation Probability: {prob_canceled}"
            ),


            html.H4(
                f"Check Out Probability: {prob_check_out}"
            ),

              html.H4(
                f"No Show Probability: {prob_no_show}"
            ),          

            html.H4(
                f"Service Level Risk: {sla_risk}"
            ),


            html.P(
                comment
            )

        ])

