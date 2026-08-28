# =========================
# src/data_loader.py
# =========================

import os
import glob
import sqlite3
import pandas as pd
import json

from datetime import datetime

from config import CONFIG


# =========================================================
# DATABASE TIMESTAMP
# =========================================================

def get_db_timestamp(db_path):

    return os.path.getmtime(
        db_path
    )


# =========================================================
# GET LATEST TRAINING DB
# =========================================================

def get_latest_db():

    db_dir = CONFIG["db_dir"]

    db_files = glob.glob(
        os.path.join(
            db_dir,
            "*.db"
        )
    )

    if not db_files:

        raise ValueError(
            f"No training database found in: "
            f"{db_dir}"
        )

    latest_db = max(
        db_files,
        key=os.path.getmtime
    )

    return latest_db


# =========================================================
# TRAINING CHECK
# =========================================================

def need_retrain(db_path):

    metadata_path = CONFIG["metadata_path"]

    # -----------------------------------------------------
    # Metadata does not exist
    # -----------------------------------------------------

    if not os.path.exists(
        metadata_path
    ):

        print(
            "[INFO] Training metadata not found."
        )

        return True

    # -----------------------------------------------------
    # Load metadata
    # -----------------------------------------------------

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as f:

        metadata = json.load(
            f
        )

    # -----------------------------------------------------
    # Current database information
    # -----------------------------------------------------

    current_db_file = os.path.basename(
        db_path
    )

    current_db_timestamp = get_db_timestamp(
        db_path
    )

    # -----------------------------------------------------
    # Metadata information
    # -----------------------------------------------------

    old_db_file = metadata.get(
        "db_file"
    )

    old_db_timestamp = metadata.get(
        "db_timestamp"
    )

    # -----------------------------------------------------
    # CHECK DB FILE NAME
    # -----------------------------------------------------

    if old_db_file != current_db_file:

        print(
            "[INFO] Training database file changed."
        )

        print(
            f"[INFO] Metadata DB : "
            f"{old_db_file}"
        )

        print(
            f"[INFO] Current DB  : "
            f"{current_db_file}"
        )

        return True

    # -----------------------------------------------------
    # CHECK DB TIMESTAMP
    # -----------------------------------------------------

    if old_db_timestamp != current_db_timestamp:

        print(
            "[INFO] Training database timestamp changed."
        )

        print(
            f"[INFO] Metadata timestamp : "
            f"{old_db_timestamp}"
        )

        print(
            f"[INFO] Current timestamp  : "
            f"{current_db_timestamp}"
        )

        return True

    # -----------------------------------------------------
    # NO CHANGE
    # -----------------------------------------------------

    print(
        "[INFO] Training database unchanged."
    )

    print(
        f"[INFO] DB file      : "
        f"{current_db_file}"
    )

    print(
        f"[INFO] DB timestamp : "
        f"{current_db_timestamp}"
    )

    return False


# =========================================================
# SAVE TRAINING METADATA
# =========================================================

def save_training_metadata(db_path):

    metadata = {

        "db_file":
            os.path.basename(
                db_path
            ),

        "db_timestamp":
            get_db_timestamp(
                db_path
            ),

        "trained_time":
            str(
                datetime.now()
            )
    }

    with open(
        CONFIG["metadata_path"],
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )


# =========================================================
# AUTO TABLE DETECTION
# =========================================================

def get_first_table(conn):

    tables = pd.read_sql(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table';
        """,
        conn
    )

    if tables.empty:

        raise ValueError(
            "No tables found in database."
        )

    return tables.iloc[0, 0]


# =========================================================
# GET ALL TABLES
# =========================================================

def get_all_tables(conn):

    tables = pd.read_sql(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table';
        """,
        conn
    )

    if tables.empty:

        raise ValueError(
            "No tables found in database."
        )

    return tables[
        "name"
    ].tolist()


# =========================================================
# LOAD & MERGE ALL TABLES
# =========================================================

def load_and_merge_tables(
    conn,
    join_key
):

    tables = get_all_tables(
        conn
    )

    print(
        f"[INFO] Found tables: "
        f"{tables}"
    )

    dfs = []

    for table in tables:

        print(
            f"[INFO] Loading table "
            f"'{table}'"
        )

        df = pd.read_sql(
            f'SELECT * FROM "{table}"',
            conn
        )

        print(
            f"[INFO] Shape: "
            f"{df.shape}"
        )

        dfs.append(
            df
        )

    # -----------------------------------------------------
    # ONLY ONE TABLE
    # -----------------------------------------------------

    if len(dfs) == 1:

        return dfs[0]

    # -----------------------------------------------------
    # VERIFY JOIN KEY
    # -----------------------------------------------------

    for i, df in enumerate(
        dfs
    ):

        if join_key not in df.columns:

            raise ValueError(
                f"Join key '{join_key}' "
                f"not found in table "
                f"'{tables[i]}'.\n"
                f"Columns available: "
                f"{df.columns.tolist()}"
            )

    # -----------------------------------------------------
    # MERGE
    # -----------------------------------------------------

    merged_df = dfs[0]

    for df in dfs[1:]:

        merged_df = pd.merge(
            merged_df,
            df,
            on=join_key,
            how="outer"
        )

    print(
        f"[INFO] Merged tables shape: "
        f"{merged_df.shape}"
    )

    return merged_df


# =========================================================
# LOAD SINGLE DB
# =========================================================

def load_single_db(
    db_path,
    join_key
):

    if (
        not db_path
        or not os.path.exists(
            db_path
        )
    ):

        raise ValueError(
            f"Invalid db_path: "
            f"{db_path}"
        )

    print(
        f"[INFO] Loading database: "
        f"{db_path}"
    )

    conn = sqlite3.connect(
        db_path
    )

    try:

        df = load_and_merge_tables(
            conn,
            join_key
        )

    finally:

        conn.close()

    return df


# =========================================================
# LOAD MULTIPLE TRAINING DBS
# =========================================================

def load_multiple_dbs(
    db_dir,
    join_key
):

    db_files = glob.glob(
        os.path.join(
            db_dir,
            "*.db"
        )
    )

    if not db_files:

        raise ValueError(
            f"No .db files found in "
            f"{db_dir}"
        )

    all_dfs = []

    for db_path in db_files:

        print(
            f"\n[INFO] Loading "
            f"{db_path}"
        )

        df = load_single_db(
            db_path,
            join_key
        )

        df[
            "source_db"
        ] = os.path.basename(
            db_path
        )

        all_dfs.append(
            df
        )

    if not all_dfs:

        raise ValueError(
            "No data loaded from DBs"
        )

    merged_df = pd.concat(
        all_dfs,
        ignore_index=True
    )

    print(
        f"[INFO] Merged "
        f"{len(db_files)} databases"
    )

    print(
        f"[INFO] Final shape: "
        f"{merged_df.shape}"
    )

    return merged_df


# =========================================================
# MAIN TRAINING / RAW DATABASE LOADER
# =========================================================

def load_data(
    config,
    db_path=None
):

    train_path = config[
        "train_data"
    ]

    # =====================================================
    # EXPLICIT DATABASE REQUEST
    # =====================================================
    #
    # This is used by app.py for dashboard loading.
    #
    # When db_path is supplied:
    #
    #     DO NOT load train.parquet
    #
    #     Load the specified RAW database instead.
    #
    # This prevents a saved processed dataframe from being
    # accidentally passed into DataProcessor.transform().
    #
    # =====================================================

    if db_path is not None:

        print(
            "[INFO] Explicit database path supplied."
        )

        print(
            "[INFO] Bypassing saved training dataset."
        )

        print(
            f"[INFO] Raw database selected: "
            f"{db_path}"
        )

        join_key = config.get(
            "join_key",
            "delivery_id"
        )

        # -------------------------------------------------
        # Load ONLY the selected database
        # -------------------------------------------------

        df = load_single_db(
            db_path,
            join_key
        )

        # -------------------------------------------------
        # Add source database
        # -------------------------------------------------

        if "source_db" not in df.columns:

            df[
                "source_db"
            ] = os.path.basename(
                db_path
            )

        # -------------------------------------------------
        # Remove duplicates
        # -------------------------------------------------

        return remove_duplicates(
            df
        )

    # =====================================================
    # EXISTING TRAINING LOADING LOGIC
    # =====================================================

    latest_db = get_latest_db()

    # -----------------------------------------------------
    # USE EXISTING PROCESSED TRAINING DATA
    # -----------------------------------------------------

    if (
        os.path.exists(
            train_path
        )
        and
        not need_retrain(
            latest_db
        )
    ):

        print(
            "[INFO] Loading trained dataset"
        )

        return pd.read_parquet(
            train_path
        )

    # -----------------------------------------------------
    # LOAD RAW TRAINING DATABASE
    # -----------------------------------------------------

    print(
        "[INFO] Loading raw training database"
    )

    join_key = config.get(
        "join_key",
        "delivery_id"
    )

    if config["multi_db"]:

        df = load_multiple_dbs(
            config["db_dir"],
            join_key
        )

    else:

        df = load_single_db(
            config["db_path"],
            join_key
        )

    return remove_duplicates(
        df
    )


# =========================================================
# LOAD PREDICTION DATABASES
# =========================================================

def load_prediction_data(
    config
):

    prediction_dir = config.get(
        "prediction_dir"
    )

    prediction_db = config.get(
        "prediction_db"
    )

    join_key = config.get(
        "join_key",
        "delivery_id"
    )

    # -----------------------------------------------------
    # OPTION 1:
    # prediction_dir = data/prediction/
    # -----------------------------------------------------

    if prediction_dir:

        db_files = glob.glob(
            os.path.join(
                prediction_dir,
                "*.db"
            )
        )

        if not db_files:

            print(
                "[INFO] No prediction "
                "databases found."
            )

            return None

        all_dfs = []

        for db_path in db_files:

            print(
                f"\n[INFO] Loading prediction DB:"
                f" {db_path}"
            )

            df = load_single_db(
                db_path,
                join_key
            )

            if df.empty:

                continue

            df[
                "source_db"
            ] = os.path.basename(
                db_path
            )

            all_dfs.append(
                df
            )

        if not all_dfs:

            print(
                "[INFO] Prediction databases "
                "contain no data."
            )

            return None

        df = pd.concat(
            all_dfs,
            ignore_index=True
        )

    # -----------------------------------------------------
    # OPTION 2:
    # single prediction DB
    # -----------------------------------------------------

    elif prediction_db:

        if not os.path.exists(
            prediction_db
        ):

            print(
                "[INFO] No prediction "
                "database found."
            )

            return None

        df = load_single_db(
            prediction_db,
            join_key
        )

    else:

        raise ValueError(
            "CONFIG must contain either "
            "'prediction_dir' or "
            "'prediction_db'."
        )

    # -----------------------------------------------------
    # EMPTY CHECK
    # -----------------------------------------------------

    if df.empty:

        print(
            "[INFO] Prediction data is empty."
        )

        return None

    # -----------------------------------------------------
    # REMOVE DUPLICATES
    # -----------------------------------------------------

    df = remove_duplicates(
        df
    )

    print(
        f"[INFO] Prediction data loaded:"
        f" {df.shape}"
    )

    return df


# =========================================================
# REMOVE DUPLICATES
# =========================================================

def remove_duplicates(
    df
):

    key_cols = (
        df.columns[2:16]
        .tolist()
    )

    if len(key_cols) >= 2:

        df = df.drop_duplicates(
            subset=key_cols,
            keep="first"
        )

    else:

        df = df.drop_duplicates()

    print(
        f"[INFO] After deduplication:"
        f" {df.shape}"
    )

    return df

