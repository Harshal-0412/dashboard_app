import io

import pandas as pd

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from .utils import clean_for_json
from .data_profiler import profile_dataframe
from .gemini_service import analyze_dataset
from .prompts import create_analysis_prompt


app = FastAPI(
    title="AI Data Analysis API",
    description="AI-powered CSV understanding and visualization API",
    version="1.0.0"
)


@app.get("/")
def root():

    return {
        "message": "AI Data Analysis API is running"
    }


@app.post("/analyze")
async def analyze_csv(
    file: UploadFile = File(...)
):

    # -----------------------------------------
    # Validate file
    # -----------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No file was provided."
        )

    if not file.filename.lower().endswith(".csv"):

        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported."
        )

    try:

        # -----------------------------------------
        # Read CSV
        # -----------------------------------------

        contents = await file.read()

        df = pd.read_csv(
            io.BytesIO(contents)
        )

        # -----------------------------------------
        # Validate dataframe
        # -----------------------------------------

        if df.empty:

            raise HTTPException(
                status_code=400,
                detail="The uploaded CSV is empty."
            )

        # -----------------------------------------
        # Generate data profile
        # -----------------------------------------

        profile = profile_dataframe(df)

        # -----------------------------------------
        # Create internal prompt
        # -----------------------------------------

        prompt = create_analysis_prompt(
            profile
        )

        # -----------------------------------------
        # Send profile to Gemini
        # -----------------------------------------

        analysis = analyze_dataset(
            profile,
            prompt
        )

        # -----------------------------------------
        # Return response
        # -----------------------------------------

        return {
            "success": True,
            "filename": file.filename,
            "profile": clean_for_json(profile),
            "analysis": clean_for_json(analysis)
        }

    except pd.errors.EmptyDataError:

        raise HTTPException(
            status_code=400,
            detail="The CSV file does not contain any data."
        )

    except pd.errors.ParserError:

        raise HTTPException(
            status_code=400,
            detail="Unable to parse the CSV file."
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )