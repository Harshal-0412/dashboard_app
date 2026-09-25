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

    # Read file for preview
    df = pd.read_csv(
        uploaded_file
    )

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    # -----------------------------------------
    # Dataset preview
    # -----------------------------------------

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    # -----------------------------------------
    # Dataset metrics
    # -----------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Rows",
            df.shape[0]
        )

    with col2:

        st.metric(
            "Columns",
            df.shape[1]
        )

    with col3:

        st.metric(
            "Missing Values",
            int(
                df.isna()
                .sum()
                .sum()
            )
        )

    # -----------------------------------------
    # Analyze button
    # -----------------------------------------

    if st.button(
        "🤖 Analyze Dataset",
        type="primary"
    ):

        with st.spinner(
            "AI is understanding your dataset..."
        ):

            uploaded_file.seek(0)

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file,
                    "text/csv"
                )
            }

            try:

                response = requests.post(
                    f"{BACKEND_URL}/analyze",
                    files=files,
                    timeout=120
                )

                if response.status_code == 200:

                    result = response.json()

                    st.session_state[
                        "analysis"
                    ] = result

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

                    st.error(
                        error_message
                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Unable to connect to FastAPI backend. "
                    "Make sure the backend is running."
                )

            except requests.exceptions.Timeout:

                st.error(
                    "The request timed out. "
                    "Please try again."
                )


# -----------------------------------------
# Display analysis
# -----------------------------------------

if "analysis" in st.session_state:

    result = st.session_state[
        "analysis"
    ]

    profile = result[
        "profile"
    ]

    analysis = result[
        "analysis"
    ]

    st.divider()

    st.header(
        "🤖 AI Dataset Understanding"
    )

    # -----------------------------------------
    # Dataset understanding
    # -----------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Dataset Type"
        )

        st.info(
            analysis[
                "dataset_type"
            ]
        )

    with col2:

        st.subheader(
            "Dataset Description"
        )

        st.write(
            analysis[
                "dataset_description"
            ]
        )

    # -----------------------------------------
    # Important columns
    # -----------------------------------------

    st.subheader(
        "Important Columns"
    )

    important_columns = analysis.get(
        "important_columns",
        []
    )

    for item in important_columns:

        st.write(
            f"**{item['column']}** — "
            f"{item['reason']}"
        )

    # -----------------------------------------
    # Visualizations
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

        st.subheader(
            f"{index + 1}. {chart['title']}"
        )

        chart_type = chart.get(
            "chart_type"
        )

        x_column = chart.get(
            "x_column"
        )

        y_column = chart.get(
            "y_column"
        )

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
                            x_column
                        )
                        .size()
                        .reset_index(
                            name="count"
                        )
                    )

                    fig = px.bar(
                        chart_df,
                        x=x_column,
                        y="count",
                        title=chart["title"]
                    )

                else:

                    chart_df = (
                        df.groupby(
                            x_column
                        )[y_column]
                        .agg(
                            aggregation
                        )
                        .reset_index()
                    )

                    fig = px.bar(
                        chart_df,
                        x=x_column,
                        y=y_column,
                        title=chart["title"]
                    )

            # ---------------------------------
            # LINE
            # ---------------------------------

            elif chart_type == "line":

                chart_df = (
                    df.groupby(
                        x_column
                    )[y_column]
                    .agg(
                        aggregation
                    )
                    .reset_index()
                )

                fig = px.line(
                    chart_df,
                    x=x_column,
                    y=y_column,
                    title=chart["title"]
                )

            # ---------------------------------
            # SCATTER
            # ---------------------------------

            elif chart_type == "scatter":

                fig = px.scatter(
                    df,
                    x=x_column,
                    y=y_column,
                    title=chart["title"]
                )

            # ---------------------------------
            # HISTOGRAM
            # ---------------------------------

            elif chart_type == "histogram":

                fig = px.histogram(
                    df,
                    x=x_column,
                    title=chart["title"]
                )

            # ---------------------------------
            # PIE
            # ---------------------------------

            elif chart_type == "pie":

                chart_df = (
                    df[x_column]
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
                    title=chart["title"]
                )

            else:

                st.warning(
                    f"Unsupported chart type: "
                    f"{chart_type}"
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
                f"Unable to create "
                f"'{chart['title']}': {str(e)}"
            )