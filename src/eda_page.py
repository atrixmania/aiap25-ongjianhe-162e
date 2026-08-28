# =========================================================
# src/eda_page.py
# =========================================================

import pandas as pd
import numpy as np
import plotly.express as px

from dash import html, dcc, Input, Output

from preprocess import DataProcessor


print("Load EDA Page")


# =========================================================
# REVERSE MAPPINGS
# =========================================================

REVERSE_MAPPINGS = DataProcessor.REVERSE_MAPPINGS


# =========================================================
# PREPARE EDA DATA
# =========================================================
#
# IMPORTANT:
#
# app.py has already done:
#
#     DataProcessor.transform()
#     FeatureEngineer.transform()
#
# Therefore this function ONLY prepares display labels.
#
# It does NOT:
#     - fit anything
#     - transform anything
#     - engineer features
#     - modify df.attrs
#
# =========================================================

def prepare_eda_data(df):

    df = df.copy()

    # =====================================================
    # SERVICE LABEL
    # =====================================================

    if "final_rating" in df.columns:

        df["service_label"] = df[
            "final_rating"
        ].apply(
            lambda x:
                "Bad"
                if x in [0, 1, 2]
                else "Neutral"
                if x == 3
                else "Good"
                if x in [4, 5, 6]
                else "Unknown"
        )

    else:

        df["service_label"] = "Unknown"

    # =====================================================
    # ENCODE LABELS
    # =====================================================

    for col, mapping in REVERSE_MAPPINGS.items():

        if col in df.columns:

            df[col + "_label"] = (
                df[col]
                .map(mapping)
                .fillna("Unknown")
            )

    return df


# =========================================================
# CREATE EDA LAYOUT
# =========================================================

def create_layout(df):

    # -----------------------------------------------------
    # Prepare display dataframe
    # -----------------------------------------------------

    df = prepare_eda_data(df)

    # -----------------------------------------------------
    # Branch options
    # -----------------------------------------------------

    if "hotel_label" in df.columns:

        branch_options = [
            {
                "label": str(b),
                "value": b
            }
            for b in
            df["hotel_label"]
            .dropna()
            .unique()
        ]

    else:

        branch_options = []

    # =====================================================
    # LAYOUT
    # =====================================================

    return html.Div([

        # =================================================
        # TITLE
        # =================================================

        html.H1(
            "Service Level Dashboard",
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
                    "⬅ Back to Main",
                    style={
                        "fontSize": "16px",
                        "padding": "10px 20px"
                    }
                ),
                href="/"
            )

        ]),

        # =================================================
        # KPI
        # =================================================

        html.Div(
            id="kpis",
            style={
                "display": "flex",
                "justifyContent": "space-around",
                "marginBottom": "20px"
            }
        ),

        # =================================================
        # FILTERS
        # =================================================

        html.Div([

            # -------------------------------------------------
            # TARGET TOGGLE
            # -------------------------------------------------

            dcc.Dropdown(
                id='target_filter',
                options=[
                    {
                        'label': 'Final Rating',
                        'value': 'final_rating'
                    },
                    {
                        'label': 'Reservation Status',
                        'value': 'reservation_status_label'
                    }
                ],
                value='final_rating',
                clearable=False
            ),

            # -------------------------------------------------
            # SERVICE FILTER
            # -------------------------------------------------

            dcc.Dropdown(
                id="service_filter",

                options=[

                    {
                        "label": "All",
                        "value": "All"
                    },

                    {
                        "label": "Good (4-6)",
                        "value": "Good"
                    },

                    {
                        "label": "Neutral (3)",
                        "value": "Neutral"
                    },

                    {
                        "label": "Bad (0-2)",
                        "value": "Bad"
                    }

                ],

                value="All"
            ),

            # -------------------------------------------------
            # BRANCH FILTER
            # -------------------------------------------------

            dcc.Dropdown(
                id="branch_filter",

                options=branch_options,

                multi=True,

                placeholder="Filter by Branch"
            )

        ], style={
            "marginBottom": "30px"
        }),

        # =================================================
        # CHARTS
        # =================================================

        dcc.Graph(id='rating_chart'),
        dcc.Graph(id='service_by_hotel'),
        dcc.Graph(id='service_level_hotel'),
        dcc.Graph(id='service_level_meal'),
        dcc.Graph(id='service_level_market_segment'),
        dcc.Graph(id='service_level_distribution_channel'),
        dcc.Graph(id='service_level_deposit_type'),
        dcc.Graph(id='service_level_customer_type'),
        dcc.Graph(id='service_level_season_group'),
        dcc.Graph(id='service_level_country_group'),
        dcc.Graph(id='service_level_adr_group'),
        dcc.Graph(id='service_level_previous_cancellations_group'),
        dcc.Graph(id='service_level_days_in_waiting_list_group'),
        dcc.Graph(id='service_level_guest_group'),
        dcc.Graph(id='service_level_is_top_bad_client'),
        dcc.Graph(id='customer_comment')

    ])


# =========================================================
# CALLBACKS
# =========================================================

def register_callbacks(app, df):

    # =====================================================
    # PREPARE ONCE
    # =====================================================

    df = prepare_eda_data(
        df
    )

    print(
        f"[INFO] EDA dataframe ready: {df.shape}"
    )

    # =====================================================
    # CALLBACK
    # =====================================================

    @app.callback(

        Output('kpis', 'children'),
        Output('rating_chart', 'figure'),
        Output('service_by_hotel', 'figure'),
        Output('service_level_hotel', 'figure'),
        Output('service_level_meal', 'figure'),
        Output('service_level_market_segment', 'figure'),
        Output('service_level_distribution_channel', 'figure'),
        Output('service_level_deposit_type', 'figure'),
        Output('service_level_customer_type', 'figure'),
        Output('service_level_season_group', 'figure'),
        Output('service_level_country_group', 'figure'),
        Output('service_level_adr_group', 'figure'),
        Output('service_level_previous_cancellations_group', 'figure'),
        Output('service_level_days_in_waiting_list_group', 'figure'),
        Output('service_level_guest_group', 'figure'),
        Output('service_level_is_top_bad_client', 'figure'),
        Output('customer_comment', 'figure'),

        Input('target_filter', 'value'),
        Input('service_filter', 'value'),
        Input('branch_filter', 'value')
    )

    def update_dashboard(
        target_filter,
        service_filter,
        branch_filter
    ):

        # =================================================
        # FILTER DATA
        # =================================================

        dff = df.copy()

        # -------------------------------------------------
        # TARGET
        # -------------------------------------------------

        target_column = target_filter

        # Safety check
        if target_column not in dff.columns:

            target_column = "final_rating"

        # -------------------------------------------------
        # SERVICE FILTER
        # -------------------------------------------------

        if (
            service_filter
            and service_filter != "All"
        ):

            dff = dff[
                dff["service_label"]
                == service_filter
            ]

        # -------------------------------------------------
        # BRANCH FILTER
        # -------------------------------------------------

        if branch_filter:

            if "hotel_label" in dff.columns:

                dff = dff[
                    dff[
                        "hotel_label"
                    ].isin(
                        branch_filter
                    )
                ]

        # =================================================
        # KPI
        # =================================================

        total = len(dff)

        if (
            total > 0
            and
            "final_rating" in dff.columns
        ):

            avg_rating = dff[
                "final_rating"
            ].mean()

        else:

            avg_rating = 0

        good = len(
            dff[
                dff["service_label"]
                == "Good"
            ]
        )

        neutral = len(
            dff[
                dff["service_label"]
                == "Neutral"
            ]
        )

        bad = len(
            dff[
                dff["service_label"]
                == "Bad"
            ]
        )

        kpis = [

            html.Div(
                f"Total: {total}"
            ),

            html.Div(
                f"Avg Rating: {avg_rating:.2f}"
            ),

            html.Div(
                f"Good: {good}"
            ),

            html.Div(
                f"Neutral: {neutral}"
            ),

            html.Div(
                f"Bad: {bad}"
            )

        ]

        # =================================================
        # TARGET DISTRIBUTION
        # =================================================

        if target_column == "final_rating":

            rating_counts = (
                dff["final_rating"]
                .dropna()
                .value_counts()
                .reindex(
                    [0, 1, 2, 3, 4, 5, 6],
                    fill_value=0
                )
                .reset_index()
            )

            rating_counts.columns = [
                "final_rating",
                "count"
            ]

            fig1 = px.pie(
                rating_counts,
                names="final_rating",
                values="count",
                title="Final Rating Distribution",
                hole=0.4
            )

        else:

            target_counts = (
                dff[target_column]
                .dropna()
                .value_counts()
                .reset_index()
            )

            target_counts.columns = [
                target_column,
                "count"
            ]

            fig1 = px.pie(
                target_counts,
                names=target_column,
                values="count",
                title="Reservation Status Distribution",
                hole=0.4
            )

        fig1.update_traces(
            textinfo="percent+label"
        )

        # =================================================
        # SERVICE BY BRANCH
        # =================================================

        if (
            "hotel_label" in dff.columns
            and
            target_column in dff.columns
        ):

            branch_df = (
                dff
                .groupby(
                    [
                        "hotel_label",
                        target_column
                    ]
                )
                .size()
                .reset_index(
                    name="count"
                )
            )

        else:

            branch_df = pd.DataFrame(
                columns=[
                    "hotel_label",
                    target_column,
                    "count"
                ]
            )

        fig2 = px.bar(
            branch_df,
            x="hotel_label",
            y="count",
            color=target_column,
            barmode="group",
            title=(
                "Final Rating by Hotel"
                if target_column == "final_rating"
                else "Reservation Status by Hotel"
            )
        )

        # =================================================
        # HELPER FUNCTION
        # =================================================

        def safe_histogram(
            data,
            color_column,
            title
        ):

            if (
                target_column not in data.columns
                or
                color_column not in data.columns
            ):

                return px.bar(
                    title=title
                )

            return px.histogram(
                data,
                x=target_column,
                color=color_column,
                barmode="group",
                title=title
            )

        # =================================================
        # SERVICE LEVEL CHARTS
        # =================================================

        service_level_hotel = safe_histogram(
            dff,
            "hotel_label",
            "Service Level by hotel"
        )

        service_level_meal = safe_histogram(
            dff,
            "meal_label",
            "Service Level by meal"
        )

        service_level_market_segment = safe_histogram(
            dff,
            "market_segment_label",
            "Service Level by market segment"
        )

        service_level_distribution_channel = safe_histogram(
            dff,
            "distribution_channel_label",
            "Service Level by distribution channel"
        )

        service_level_deposit_type = safe_histogram(
            dff,
            "deposit_type_group",
            "Service Level by deposit type"
        )

        service_level_customer_type = safe_histogram(
            dff,
            "customer_type",
            "Service Level by customer type"
        )

        service_level_season_group = safe_histogram(
            dff,
            "season_group",
            "Service Level by season"
        )

        service_level_country_group = safe_histogram(
            dff,
            "country_group",
            "Service Level by country"
        )

        service_level_adr_group = safe_histogram(
            dff,
            "adr_group",
            "Service Level by average daily rate"
        )

        # FIXED COLUMN NAME
        service_level_previous_cancellations_group = safe_histogram(
            dff,
            "previous_cancellations_group",
            "Service Level by previous cancellations"
        )

        service_level_days_in_waiting_list_group = safe_histogram(
            dff,
            "days_in_waiting_list_group",
            "Service Level by days in waiting list"
        )

        service_level_guest_group = safe_histogram(
            dff,
            "guest_group",
            "Service Level by guest group"
        )

        # =================================================
        # TOP CLIENT FILTER
        # =================================================

        if service_filter == "Bad":

            rating_mask = [0, 1, 2]

        elif service_filter == "Neutral":

            rating_mask = [3]

        elif service_filter == "Good":

            rating_mask = [4, 5, 6]

        else:

            rating_mask = [
                0,
                1,
                2,
                3,
                4,
                5,
                6
            ]

        # =================================================
        # TOP CLIENTS
        # =================================================

        if (
            "client_email" in dff.columns
            and
            "final_rating" in dff.columns
        ):

            top_clients = (
                dff[
                    dff[
                        "final_rating"
                    ].isin(
                        rating_mask
                    )
                ]
                .groupby(
                    "client_email"
                )
                .size()
                .reset_index(
                    name="count"
                )
                .sort_values(
                    "count",
                    ascending=False
                )
                .head(10)
            )

        else:

            top_clients = pd.DataFrame(
                columns=[
                    "client_email",
                    "count"
                ]
            )

        fig_top_clients = px.bar(
            top_clients,
            x="client_email",
            y="count",
            title=(
                "Top 10 Clients "
                "(Filtered Ratings)"
            ),
            text="count"
        )

        fig_top_clients.update_layout(
            xaxis_title="client_email",
            yaxis_title="Count of Ratings"
        )

        # =================================================
        # BUSINESS CATEGORY SUMMARY
        # =================================================
        #
        # IMPORTANT:
        #
        # FeatureEngineer provides:
        #
        #     client_email
        #     business_categories
        #
        # The same client may appear in many booking rows.
        #
        # Therefore:
        #
        #     DO NOT count every row.
        #
        # Instead:
        #
        #     1. Keep client_email
        #     2. Explode business_categories
        #     3. Remove duplicate
        #        client_email + category
        #        combinations
        #     4. Count unique clients per category
        #
        # Example:
        #
        # Client A -> Cleanliness
        # Client A -> Cleanliness
        # Client A -> Cleanliness
        #
        # becomes:
        #
        # Client A -> Cleanliness
        #
        # and therefore counts as 1 client.
        #
        # =================================================

        if (
            "client_email" in dff.columns
            and
            "business_categories" in dff.columns
        ):

            df_business = (
                dff[
                    [
                        "client_email",
                        "business_categories"
                    ]
                ]
                .dropna(
                    subset=[
                        "client_email",
                        "business_categories"
                    ]
                )
                .explode(
                    "business_categories"
                )
            )

            # Remove empty categories
            df_business = df_business[
                df_business[
                    "business_categories"
                ].notna()
            ]

            # -------------------------------------------------
            # IMPORTANT:
            #
            # One client/category combination = one count
            #
            # This prevents repeated vlookup values from
            # inflating the business-category count.
            # -------------------------------------------------

            df_business = (
                df_business[
                    [
                        "client_email",
                        "business_categories"
                    ]
                ]
                .drop_duplicates(
                    subset=[
                        "client_email",
                        "business_categories"
                    ]
                )
            )

            # -------------------------------------------------
            # Count unique clients for each category
            # -------------------------------------------------

            business_summary = (
                df_business
                .groupby(
                    "business_categories"
                )[
                    "client_email"
                ]
                .nunique()
                .reset_index(
                    name="count"
                )
                .sort_values(
                    "count",
                    ascending=False
                )
            )

        elif "business_categories" in dff.columns:

            # -------------------------------------------------
            # Fallback if client_email is unavailable
            # -------------------------------------------------

            df_business = (
                dff[
                    [
                        "business_categories"
                    ]
                ]
                .explode(
                    "business_categories"
                )
                .dropna(
                    subset=[
                        "business_categories"
                    ]
                )
            )

            business_summary = (
                df_business[
                    "business_categories"
                ]
                .value_counts()
                .reset_index()
            )

            business_summary.columns = [
                "business_categories",
                "count"
            ]

        else:

            business_summary = pd.DataFrame(
                columns=[
                    "business_categories",
                    "count"
                ]
            )

        # =================================================
        # BUSINESS CATEGORY CHART
        # =================================================

        fig_business_global = px.bar(
            business_summary,
            x="business_categories",
            y="count",
            title="Customer Feedback by Business Category",
            text="count"
        )

        fig_business_global.update_layout(
            xaxis_title="Business Category",
            yaxis_title="Unique Clients"
        )

        # =================================================
        # RETURN ALL OUTPUTS
        # =================================================

        return (

            kpis,
            fig1,
            fig2,
            service_level_hotel,
            service_level_meal,
            service_level_market_segment,
            service_level_distribution_channel,
            service_level_deposit_type,
            service_level_customer_type,
            service_level_season_group,
            service_level_country_group,
            service_level_adr_group,
            service_level_previous_cancellations_group,
            service_level_days_in_waiting_list_group,
            service_level_guest_group,
            fig_top_clients,
            fig_business_global

        )



