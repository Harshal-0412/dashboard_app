import requests
import pandas as pd
import streamlit as st
import plotly.express as px


BACKEND_URL = "http://127.0.0.1:8000"


# -----------------------------------------
# Page configuration
# -----------------------------------------

st.set_page_config(
    page_title="AI Data Dashboard",
    page_icon="📊",
    layout="wide"
)


# -----------------------------------------
# Helper functions
# -----------------------------------------

@st.cache_data(show_spinner=False)
def load_csv(file_bytes):
    """Read the CSV once and cache it."""
    from io import BytesIO
    return pd.read_csv(BytesIO(file_bytes))


def create_heatmap(dataframe):
    """Create a correlation heatmap for numeric columns."""
    numeric_df = dataframe.select_dtypes(include=["number"])

    if numeric_df.shape[1] < 2:
        return None

    # Correlation is calculated on valid numeric observations.
    corr_matrix = numeric_df.corr(numeric_only=True)

    # Remove columns that have no variation / no usable correlation.
    corr_matrix = corr_matrix.dropna(axis=0, how="all")
    corr_matrix = corr_matrix.dropna(axis=1, how="all")

    if corr_matrix.shape[0] < 2:
        return None

    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        origin="lower",
        title="Correlation Between Numerical Features"
    )

    fig.update_layout(
        height=650,
        xaxis_title="Features",
        yaxis_title="Features",
        coloraxis_colorbar=dict(title="Correlation")
    )

    return fig


def validate_chart_columns(dataframe, chart):
    """Make sure Gemini did not return columns that do not exist."""
    available_columns = set(dataframe.columns)

    x_column = chart.get("x_column")
    y_column = chart.get("y_column")

    if x_column and x_column not in available_columns:
        return False

    if y_column and y_column not in available_columns:
        return False

    return True


# -----------------------------------------
# Header
# -----------------------------------------

st.title("📊 AI Data Dashboard")

st.write(
    "Upload a CSV file and let AI understand "
    "your data and automatically generate "
    "useful visualizations."
)


# -----------------------------------------
# File upload
# -----------------------------------------

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)


if uploaded_file is not None:

    # Read only once using Streamlit cache.
    file_bytes = uploaded_file.getvalue()

    try:
        df = load_csv(file_bytes)

    except Exception as e:
        st.error(f"Unable to read CSV file: {e}")
        st.stop()

    if df.empty:
        st.warning("The uploaded CSV is empty.")
        st.stop()

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    # -----------------------------------------
    # Dataset preview
    # -----------------------------------------

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        height=320,
        use_container_width=True
    )

    # -----------------------------------------
    # Dataset metrics
    # -----------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Rows",
            f"{df.shape[0]:,}"
        )

    with col2:
        st.metric(
            "Columns",
            df.shape[1]
        )

    with col3:
        st.metric(
            "Missing Values",
            f"{int(df.isna().sum().sum()):,}"
        )

    with col4:
        st.metric(
            "Duplicate Rows",
            f"{int(df.duplicated().sum()):,}"
        )

    # -----------------------------------------
    # Heatmap
    # -----------------------------------------

    st.divider()
    st.subheader("🔥 Correlation Heatmap")

    numeric_column_count = len(
        df.select_dtypes(include=["number"]).columns
    )

    if numeric_column_count >= 2:

        if st.button(
            "Generate Heatmap",
            key="generate_heatmap"
        ):

            with st.spinner(
                "Calculating correlations..."
            ):
                st.session_state["heatmap"] = (
                    create_heatmap(df)
                )

        if st.session_state.get("heatmap") is not None:

            st.plotly_chart(
                st.session_state["heatmap"],
                use_container_width=True
            )

            st.caption(
                "Values close to +1 indicate strong positive correlation, "
                "values close to -1 indicate strong negative correlation, "
                "and values close to 0 indicate weak linear correlation."
            )

    else:
        st.info(
            "Heatmap requires at least two numerical columns."
        )

    # -----------------------------------------
    # Analyze button
    # -----------------------------------------

    st.divider()

    if st.button(
        "🤖 Analyze Dataset",
        type="primary",
        key="analyze_dataset"
    ):

        with st.spinner(
            "AI is understanding your dataset..."
        ):

            try:

                # Send bytes instead of the Streamlit
                # UploadedFile object.
                files = {
                    "file": (
                        uploaded_file.name,
                        file_bytes,
                        "text/csv"
                    )
                }

                response = requests.post(
                    f"{BACKEND_URL}/analyze",
                    files=files,
                    timeout=(10, 90)
                )

                if response.status_code == 200:

                    result = response.json()

                    st.session_state["analysis"] = result

                    st.success(
                        "Dataset analysis completed!"
                    )

                else:

                    try:
                        error_message = (
                            response.json()
                            .get(
                                "detail",
                                "Something went wrong."
                            )
                        )

                    except Exception:
                        error_message = (
                            "Backend returned an error."
                        )

                    st.error(error_message)

            except requests.exceptions.ConnectTimeout:

                st.error(
                    "FastAPI did not respond within 10 seconds. "
                    "Make sure the backend is running."
                )

            except requests.exceptions.ReadTimeout:

                st.error(
                    "The AI analysis took longer than 90 seconds. "
                    "Please check the FastAPI terminal for the error."
                )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Unable to connect to FastAPI backend. "
                    "Run: uvicorn backend.main:app --reload"
                )

            except requests.exceptions.RequestException as e:

                st.error(
                    f"Request failed: {e}"
                )


# -----------------------------------------
# Display AI analysis
# -----------------------------------------

if "analysis" in st.session_state:

    result = st.session_state["analysis"]

    profile = result.get("profile", {})
    analysis = result.get("analysis", {})

    st.divider()

    st.header(
        "🤖 AI Dataset Understanding"
    )

    # -----------------------------------------
    # Dataset understanding
    # -----------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Dataset Type")

        st.info(
            analysis.get(
                "dataset_type",
                "Not available"
            )
        )

    with col2:

        st.subheader("Dataset Description")

        st.write(
            analysis.get(
                "dataset_description",
                "Not available"
            )
        )

    # -----------------------------------------
    # Important columns
    # -----------------------------------------

    st.subheader("Important Columns")

    important_columns = analysis.get(
        "important_columns",
        []
    )

    if important_columns:

        for item in important_columns:

            st.write(
                f"**{item.get('column', 'Unknown')}** — "
                f"{item.get('reason', '')}"
            )

    else:

        st.info(
            "No important columns were returned by the AI."
        )

    # -----------------------------------------
    # AI visualizations
    # -----------------------------------------

    st.divider()

    st.header(
        "📈 AI Recommended Visualizations"
    )

    visualizations = analysis.get(
        "visualizations",
        []
    )

    for index, chart in enumerate(
        visualizations
    ):

        title = chart.get(
            "title",
            f"Visualization {index + 1}"
        )

        st.subheader(
            f"{index + 1}. {title}"
        )

        # Protect the application from invalid
        # column names returned by Gemini.
        if not validate_chart_columns(
            df,
            chart
        ):
            st.warning(
                "This visualization was skipped because "
                "the AI returned a column that does not exist "
                "in the uploaded dataset."
            )
            continue

        chart_type = chart.get("chart_type")
        x_column = chart.get("x_column")
        y_column = chart.get("y_column")
        aggregation = chart.get(
            "aggregation",
            "none"
        )

        try:

            # ---------------------------------
            # BAR
            # ---------------------------------

            if chart_type == "bar":

                if aggregation == "count":

                    chart_df = (
                        df.groupby(
                            x_column,
                            dropna=False
                        )
                        .size()
                        .reset_index(
                            name="count"
                        )
                    )

                    chart_df[x_column] = (
                        chart_df[x_column]
                        .fillna("Unknown")
                        .astype(str)
                    )

                    fig = px.bar(
                        chart_df,
                        x=x_column,
                        y="count",
                        color=x_column,
                        text_auto=True,
                        color_discrete_sequence=(
                            px.colors.qualitative.Set2
                        ),
                        title=title
                    )

                else:

                    chart_df = (
                        df.groupby(
                            x_column,
                            dropna=False
                        )[y_column]
                        .agg(aggregation)
                        .reset_index()
                    )

                    chart_df[x_column] = (
                        chart_df[x_column]
                        .fillna("Unknown")
                        .astype(str)
                    )

                    fig = px.bar(
                        chart_df,
                        x=x_column,
                        y=y_column,
                        color=x_column,
                        text_auto=".2s",
                        color_discrete_sequence=(
                            px.colors.qualitative.Set2
                        ),
                        title=title
                    )

                fig.update_layout(
                    showlegend=False
                )

            # ---------------------------------
            # LINE
            # ---------------------------------

            elif chart_type == "line":

                chart_df = (
                    df.groupby(
                        x_column,
                        dropna=False
                    )[y_column]
                    .agg(aggregation)
                    .reset_index()
                )

                chart_df[x_column] = (
                    chart_df[x_column]
                    .fillna("Unknown")
                )

                fig = px.line(
                    chart_df,
                    x=x_column,
                    y=y_column,
                    markers=True,
                    title=title
                )

            # ---------------------------------
            # SCATTER
            # ---------------------------------

            elif chart_type == "scatter":

                categorical_cols = (
                    df.select_dtypes(
                        include=[
                            "object",
                            "category"
                        ]
                    )
                    .columns
                    .tolist()
                )

                color_col = (
                    categorical_cols[0]
                    if categorical_cols
                    else None
                )

                # Limit hover columns to avoid creating
                # an unnecessarily huge Plotly object.
                hover_columns = list(
                    df.columns[:10]
                )

                fig = px.scatter(
                    df,
                    x=x_column,
                    y=y_column,
                    color=color_col,
                    hover_data=hover_columns,
                    color_discrete_sequence=(
                        px.colors.qualitative.Bold
                    ),
                    title=title
                )

                fig.update_traces(
                    marker=dict(
                        size=7,
                        opacity=0.75
                    )
                )

            # ---------------------------------
            # HISTOGRAM
            # ---------------------------------

            elif chart_type == "histogram":

                fig = px.histogram(
                    df,
                    x=x_column,
                    nbins=40,
                    title=title
                )

                fig.update_layout(
                    bargap=0.05
                )

            # ---------------------------------
            # PIE
            # ---------------------------------

            elif chart_type == "pie":

                chart_df = (
                    df[x_column]
                    .fillna("Unknown")
                    .value_counts()
                    .reset_index()
                )

                chart_df.columns = [
                    x_column,
                    "count"
                ]

                fig = px.pie(
                    chart_df,
                    names=x_column,
                    values="count",
                    color=x_column,
                    color_discrete_sequence=(
                        px.colors.qualitative.Pastel
                    ),
                    hole=0.45,
                    title=title
                )

                fig.update_traces(
                    textposition="inside",
                    textinfo="percent+label"
                )

            else:

                st.warning(
                    f"Unsupported chart type: {chart_type}"
                )

                continue

            # ---------------------------------
            # Display chart
            # ---------------------------------

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.caption(
                chart.get(
                    "reason",
                    ""
                )
            )

        except Exception as e:

            st.warning(
                f"Unable to create '{title}': {e}"
            )
