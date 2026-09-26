import pandas as pd
import numpy as np

def profile_dataframe(df: pd.DataFrame) -> dict:
    # Create working copy
    working_df = df.copy()

    # Convert ±infinity to NaN
    working_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    profile = {
        "dataset": {
            "number_of_rows": int(working_df.shape[0]),
            "number_of_columns": int(working_df.shape[1]),
            "duplicate_rows": int(working_df.duplicated().sum()),
            "memory_usage_mb": round(
                working_df.memory_usage(deep=True).sum() / (1024 * 1024), 2
            )
        },
        "columns": [],
        "numeric_summary": {},
        "categorical_summary": {},
        "sample_data": working_df.head(5).to_dict(orient="records")
    }

    # Column metadata
    for col in working_df.columns:
        s = working_df[col]

        profile["columns"].append({
            "name": col,
            "dtype": str(s.dtype),
            "missing_values": int(s.isna().sum()),
            "missing_percentage": round((s.isna().mean()) * 100, 2),
            "unique_values": int(s.nunique(dropna=True)),
            "example_values": s.dropna().head(5).tolist()
        })

    # Numeric statistics (NaN automatically ignored)
    numeric_cols = working_df.select_dtypes(include=np.number).columns

    if len(numeric_cols):
        stats = (
            working_df[numeric_cols]
            .describe(percentiles=[0.25, 0.5, 0.75])
            .round(2)
            .to_dict()
        )

        profile["numeric_summary"] = stats

    # Categorical statistics
    cat_cols = working_df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    for col in cat_cols:
        temp = working_df[col].fillna("Unknown")

        profile["categorical_summary"][col] = (
            temp.value_counts()
            .head(10)
            .to_dict()
        )

    return profile