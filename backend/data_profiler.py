import math
from typing import Any, Optional

import numpy as np
import pandas as pd


# ============================================================
# Helper Functions
# ============================================================

def make_json_safe(value: Any) -> Any:
    """
    Convert NumPy/Pandas values into JSON-safe Python values.

    Handles:
    - NaN
    - Infinity
    - NumPy integers/floats
    - Pandas timestamps
    """

    if value is None:
        return None

    # Pandas / NumPy missing values
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    # NumPy integer
    if isinstance(value, np.integer):
        return int(value)

    # NumPy floating point
    if isinstance(value, np.floating):
        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return None

        return value

    # Python float
    if isinstance(value, float):

        if math.isnan(value) or math.isinf(value):
            return None

        return value

    # NumPy boolean
    if isinstance(value, np.bool_):
        return bool(value)

    # Timestamp
    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    # Python datetime/date
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass

    # Lists
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]

    # Dictionaries
    if isinstance(value, dict):
        return {
            str(k): make_json_safe(v)
            for k, v in value.items()
        }

    return value


def value_counts_list(series: pd.Series, limit: Optional[int] = None) -> list:
    """
    Return value/count information in decreasing frequency order.

    Missing values are represented as 'Unknown'.
    """

    counts = (
        series
        .fillna("Unknown")
        .value_counts(dropna=False)
    )

    if limit is not None:
        counts = counts.head(limit)

    result = []

    for value, count in counts.items():

        result.append({
            "value": make_json_safe(value),
            "count": int(count)
        })

    return result


# ============================================================
# Datetime Detection
# ============================================================

def is_datetime_column(
    series: pd.Series,
    threshold: float = 0.90
) -> bool:
    """
    Detect whether a column is datetime-like.

    Detection rules:
    1. Already a datetime dtype -> True
    2. Object/string column:
       - Try parsing non-null values
       - At least 90% of sampled values must be parseable

    This avoids incorrectly treating normal categorical
    columns as dates.
    """

    # Already datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    # Don't try to parse numeric columns as dates
    if pd.api.types.is_numeric_dtype(series):
        return False

    if series.dtype == "object" or pd.api.types.is_string_dtype(series):

        non_null = series.dropna()

        if len(non_null) == 0:
            return False

        # Limit parsing to avoid expensive processing
        sample = non_null.head(1000)

        try:
            parsed = pd.to_datetime(
                sample,
                errors="coerce"
            )

            valid_ratio = parsed.notna().mean()

            return bool(valid_ratio >= threshold)

        except Exception:
            return False

    return False


# ============================================================
# Categorical Column Profiling
# ============================================================

def profile_categorical_column(
    series: pd.Series
) -> dict:
    """
    Profile a categorical column.

    Includes:
    - unique_count
    - unique_percentage
    - top value
    - top value frequency
    - top value percentage
    - rare category count
    - unique values + counts

    If unique_count < 10:
        return all unique values.

    Otherwise:
        return top 5 values only.
    """

    total_count = len(series)

    non_null = series.dropna()

    unique_count = int(
        non_null.nunique()
    )

    unique_percentage = (
        (unique_count / total_count) * 100
        if total_count > 0
        else 0
    )

    counts = (
        series
        .fillna("Unknown")
        .value_counts(dropna=False)
    )

    # --------------------------------------------------------
    # Top category
    # --------------------------------------------------------

    top_value = None
    top_value_frequency = 0
    top_value_percentage = 0.0

    if not counts.empty:

        top_value = make_json_safe(
            counts.index[0]
        )

        top_value_frequency = int(
            counts.iloc[0]
        )

        if total_count > 0:

            top_value_percentage = round(
                (top_value_frequency / total_count) * 100,
                2
            )

    # --------------------------------------------------------
    # Rare categories
    # --------------------------------------------------------
    #
    # A rare category is defined as occurring in
    # less than 1% of the total records.
    #
    # Missing values ("Unknown") are excluded from this
    # calculation.
    # --------------------------------------------------------

    if total_count > 0:

        rare_threshold = total_count * 0.01

        non_null_counts = (
            non_null.value_counts()
        )

        number_of_rare_categories = int(
            (non_null_counts < rare_threshold).sum()
        )

    else:
        number_of_rare_categories = 0

    result = {

        "data_type": "categorical",

        "total_count": total_count,

        "missing_count": int(
            series.isna().sum()
        ),

        "missing_percentage": round(
            series.isna().mean() * 100,
            2
        ) if total_count > 0 else 0,

        "unique_count": unique_count,

        "unique_percentage": round(
            unique_percentage,
            2
        ),

        "top_value": top_value,

        "top_value_frequency": top_value_frequency,

        "top_value_percentage": top_value_percentage,

        "number_of_rare_categories":
            number_of_rare_categories
    }

    # --------------------------------------------------------
    # Unique values
    # --------------------------------------------------------

    if unique_count < 10:

        # Return every unique value
        result["unique_values"] = (
            value_counts_list(series)
        )

    else:

        # Return only top 5
        result["unique_values"] = (
            value_counts_list(
                series,
                limit=5
            )
        )

    return result


# ============================================================
# Numeric Column Profiling
# ============================================================

def profile_numeric_column(
    series: pd.Series
) -> dict:
    """
    Profile a numeric column.

    Includes:
    - min
    - max
    - mean
    - median
    - std
    - variance
    - Q1
    - Q3
    - IQR
    - skewness
    - zero count
    - negative count
    - outlier count
    - unique count
    - unique values when cardinality < 10
    """

    total_count = len(series)

    # Remove NaN and infinity
    clean_series = series.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    unique_count = int(
        clean_series.nunique()
    )

    result = {

        "data_type": "numeric",

        "total_count": total_count,

        "missing_count": int(
            series.isna().sum()
        ),

        "missing_percentage": round(
            series.isna().mean() * 100,
            2
        ) if total_count > 0 else 0,

        "unique_count": unique_count,

        "min": None,
        "max": None,
        "mean": None,
        "median": None,
        "std": None,
        "variance": None,

        "Q1": None,
        "Q3": None,
        "IQR": None,

        "skewness": None,

        "zero_count": int(
            (series == 0).sum()
        ),

        "negative_count": int(
            (series < 0).sum()
        ),

        "outlier_count": 0
    }

    # No valid numeric data
    if clean_series.empty:

        if unique_count < 10:

            result["unique_values"] = (
                value_counts_list(series)
            )

        return result

    # --------------------------------------------------------
    # Descriptive statistics
    # --------------------------------------------------------

    minimum = clean_series.min()
    maximum = clean_series.max()
    mean = clean_series.mean()
    median = clean_series.median()

    std = clean_series.std()
    variance = clean_series.var()

    q1 = clean_series.quantile(0.25)
    q3 = clean_series.quantile(0.75)

    iqr = q3 - q1

    # --------------------------------------------------------
    # Skewness
    # --------------------------------------------------------

    if len(clean_series) >= 3:

        skewness = clean_series.skew()

    else:

        skewness = np.nan

    # --------------------------------------------------------
    # Outliers using IQR
    # --------------------------------------------------------

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    outlier_mask = (
        (clean_series < lower_bound) |
        (clean_series > upper_bound)
    )

    outlier_count = int(
        outlier_mask.sum()
    )

    result.update({

        "min": make_json_safe(minimum),

        "max": make_json_safe(maximum),

        "mean": make_json_safe(mean),

        "median": make_json_safe(median),

        "std": make_json_safe(std),

        "variance": make_json_safe(variance),

        "Q1": make_json_safe(q1),

        "Q3": make_json_safe(q3),

        "IQR": make_json_safe(iqr),

        "skewness": make_json_safe(skewness),

        "outlier_count": outlier_count
    })

    # --------------------------------------------------------
    # Unique value distribution
    # --------------------------------------------------------

    if unique_count < 10:

        result["unique_values"] = (
            value_counts_list(series)
        )

    return result


# ============================================================
# Datetime / Timeframe Profiling
# ============================================================

def profile_datetime_column(
    series: pd.Series
) -> dict:
    """
    Profile a datetime/timeframe column.

    Includes:
    - min_date
    - max_date
    - date_range
    - number_of_unique_dates
    - missing_dates
    - frequency
    """

    # Convert to datetime
    dates = pd.to_datetime(
        series,
        errors="coerce"
    )

    total_count = len(series)

    missing_count = int(
        dates.isna().sum()
    )

    result = {

        "data_type": "datetime",

        "total_count": total_count,

        "missing_count": missing_count,

        "missing_percentage": round(
            (missing_count / total_count) * 100,
            2
        ) if total_count > 0 else 0,

        "min_date": None,

        "max_date": None,

        "date_range": None,

        "number_of_unique_dates": 0,

        "missing_dates": [],

        "frequency": "unknown"
    }

    valid_dates = dates.dropna()

    if valid_dates.empty:
        return result

    # Normalize to remove time component
    normalized_dates = (
        valid_dates
        .dt.normalize()
    )

    min_date = normalized_dates.min()
    max_date = normalized_dates.max()

    result["min_date"] = (
        min_date.strftime("%Y-%m-%d")
    )

    result["max_date"] = (
        max_date.strftime("%Y-%m-%d")
    )

    result["date_range"] = str(
        max_date - min_date
    )

    unique_dates = (
        normalized_dates
        .drop_duplicates()
        .sort_values()
    )

    result["number_of_unique_dates"] = int(
        unique_dates.nunique()
    )

    # --------------------------------------------------------
    # Detect frequency
    # --------------------------------------------------------

    if len(unique_dates) >= 2:

        differences = (
            unique_dates
            .diff()
            .dropna()
        )

        # Median interval is more robust than
        # using the first interval.
        median_difference = (
            differences.median()
        )

        if median_difference <= pd.Timedelta(hours=1):

            frequency = "hourly"

        elif median_difference <= pd.Timedelta(days=1):

            frequency = "daily"

        elif median_difference <= pd.Timedelta(days=7):

            frequency = "weekly"

        elif median_difference <= pd.Timedelta(days=31):

            frequency = "monthly"

        elif median_difference <= pd.Timedelta(days=92):

            frequency = "quarterly"

        elif median_difference <= pd.Timedelta(days=366):

            frequency = "yearly"

        else:

            frequency = "irregular"

        result["frequency"] = frequency

    elif len(unique_dates) == 1:

        result["frequency"] = "single_date"

    # --------------------------------------------------------
    # Missing dates
    # --------------------------------------------------------
    #
    # Only calculate missing dates when the frequency is
    # daily. For irregular datasets, generating an enormous
    # expected date range can be misleading.
    # --------------------------------------------------------

    if result["frequency"] == "daily":

        expected_dates = pd.date_range(
            start=min_date,
            end=max_date,
            freq="D"
        )

        existing_dates = pd.DatetimeIndex(
            unique_dates
        )

        missing_dates = (
            expected_dates
            .difference(existing_dates)
        )

        result["missing_dates"] = [
            date.strftime("%Y-%m-%d")
            for date in missing_dates
        ]

    return result


# ============================================================
# Column Profiling
# ============================================================

def profile_columns(
    df: pd.DataFrame
) -> dict:
    """
    Generate detailed profiling for every column.
    """

    column_profiles = {}

    for column in df.columns:

        series = df[column]

        # ----------------------------------------------------
        # Datetime
        # ----------------------------------------------------

        if is_datetime_column(series):

            column_profiles[column] = (
                profile_datetime_column(series)
            )

        # ----------------------------------------------------
        # Numeric
        # ----------------------------------------------------

        elif pd.api.types.is_numeric_dtype(series):

            column_profiles[column] = (
                profile_numeric_column(series)
            )

        # ----------------------------------------------------
        # Categorical
        # ----------------------------------------------------

        else:

            column_profiles[column] = (
                profile_categorical_column(series)
            )

    return column_profiles


# ============================================================
# Main DataFrame Profiler
# ============================================================

def profile_dataframe(
    df: pd.DataFrame
) -> dict:
    """
    Generate a complete production-ready profile
    for a pandas DataFrame.

    The output is designed to be passed to the
    Gemini LLM for intelligent visualization selection.
    """

    if df is None:
        raise ValueError(
            "DataFrame cannot be None."
        )

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Input must be a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "DataFrame is empty."
        )

    # --------------------------------------------------------
    # Create working copy
    # --------------------------------------------------------

    working_df = df.copy()

    # --------------------------------------------------------
    # Replace infinity with NaN
    # --------------------------------------------------------

    working_df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    # --------------------------------------------------------
    # Dataset-level information
    # --------------------------------------------------------

    dataset_profile = {

        "number_of_rows": int(
            working_df.shape[0]
        ),

        "number_of_columns": int(
            working_df.shape[1]
        ),

        "duplicate_rows": int(
            working_df.duplicated().sum()
        ),

        "memory_usage_mb": round(
            working_df.memory_usage(
                deep=True
            ).sum() / (1024 * 1024),
            2
        ),

        "total_missing_values": int(
            working_df.isna().sum().sum()
        )
    }

    # --------------------------------------------------------
    # Detailed column profiles
    # --------------------------------------------------------

    column_profiles = profile_columns(
        working_df
    )

    # --------------------------------------------------------
    # Basic column metadata
    # --------------------------------------------------------

    column_metadata = []

    for column in working_df.columns:

        series = working_df[column]

        metadata = {

            "name": str(column),

            "dtype": str(
                series.dtype
            ),

            "missing_values": int(
                series.isna().sum()
            ),

            "missing_percentage": round(
                series.isna().mean() * 100,
                2
            ),

            "unique_values": int(
                series.nunique(
                    dropna=True
                )
            ),

            "example_values": [
                make_json_safe(value)
                for value in (
                    series
                    .dropna()
                    .head(5)
                    .tolist()
                )
            ]
        }

        column_metadata.append(
            metadata
        )

    # --------------------------------------------------------
    # Sample data
    # --------------------------------------------------------

    sample_data = (
        working_df
        .head(5)
        .to_dict(orient="records")
    )

    sample_data = make_json_safe(
        sample_data
    )

    # --------------------------------------------------------
    # Final profile
    # --------------------------------------------------------

    profile = {

        "dataset": dataset_profile,

        "columns": column_metadata,

        "column_profiles": column_profiles,

        "sample_data": sample_data
    }

    # --------------------------------------------------------
    # Final JSON-safe conversion
    # --------------------------------------------------------

    return make_json_safe(
        profile
    )

# import pandas as pd
# import numpy as np
#
# def profile_dataframe(df: pd.DataFrame) -> dict:
#     # Create working copy
#     working_df = df.copy()
#
#     # Convert ±infinity to NaN
#     working_df.replace([np.inf, -np.inf], np.nan, inplace=True)
#
#     profile = {
#         "dataset": {
#             "number_of_rows": int(working_df.shape[0]),
#             "number_of_columns": int(working_df.shape[1]),
#             "duplicate_rows": int(working_df.duplicated().sum()),
#             "memory_usage_mb": round(
#                 working_df.memory_usage(deep=True).sum() / (1024 * 1024), 2
#             )
#         },
#         "columns": [],
#         "numeric_summary": {},
#         "categorical_summary": {},
#         "sample_data": working_df.head(5).to_dict(orient="records")
#     }
#
#     # Column metadata
#     for col in working_df.columns:
#         s = working_df[col]
#
#         profile["columns"].append({
#             "name": col,
#             "dtype": str(s.dtype),
#             "missing_values": int(s.isna().sum()),
#             "missing_percentage": round((s.isna().mean()) * 100, 2),
#             "unique_values": int(s.nunique(dropna=True)),
#             "example_values": s.dropna().head(5).tolist()
#         })
#
#     # Numeric statistics (NaN automatically ignored)
#     numeric_cols = working_df.select_dtypes(include=np.number).columns
#
#     if len(numeric_cols):
#         stats = (
#             working_df[numeric_cols]
#             .describe(percentiles=[0.25, 0.5, 0.75])
#             .round(2)
#             .to_dict()
#         )
#
#         profile["numeric_summary"] = stats
#
#     # Categorical statistics
#     cat_cols = working_df.select_dtypes(
#         include=["object", "category", "bool"]
#     ).columns
#
#     for col in cat_cols:
#         temp = working_df[col].fillna("Unknown")
#
#         profile["categorical_summary"][col] = (
#             temp.value_counts()
#             .head(10)
#             .to_dict()
#         )
#
#     return profile