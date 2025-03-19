# main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import logging
import os
from io import BytesIO
from dotenv import load_dotenv
from datetime import timedelta

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Import the async process_input function and Firestore client from summary.py
from summary import process_input, db
from firebase_admin import storage, firestore

app = FastAPI(
    title="Bill Summarization API",
    description="An API to generate bill summaries using GPT-4, GPT-4-mini, and TogetherAI models.",
    version="1.0.0"
)

# Mount static files from the "static" directory.
app.mount("/static", StaticFiles(directory="static"), name="static")

# Root endpoint to serve index.html
@app.get("/", response_class=HTMLResponse)
async def read_index():
    return FileResponse("static/index.html")

@app.get("/list_files")
async def list_files():
    """
    List previously uploaded files for the guest user.
    Returns a JSON object mapping file names to signed URLs.
    """
    try:
        bucket = storage.bucket()  # Use the default Firebase bucket
        prefix = "users/guest_user/"    # Default folder for guest user
        blobs = bucket.list_blobs(prefix=prefix)
        files = {}
        for blob in blobs:
            file_name = blob.name.replace(prefix, "", 1)
            if file_name:
                file_url = blob.generate_signed_url(expiration=timedelta(hours=1))
                files[file_name] = file_url
        return files
    except Exception as e:
        logging.error(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize/url")
async def summarize_url(
    url: str = Form(..., description="The URL of the document to summarize"),
    custom_prompt: str = Form(..., description="Your custom prompt for summarization"),
    override_base_name: str = Form(None, description="Optional override for base name")
):
    """
    Summarize a document provided by its URL.
    Generates three summaries: GPT-4, GPT-4-mini, and TogetherAI.
    Uses default guest values.
    """
    try:
        user_id = "guest_user"
        display_name = "Guest"
        result_gpt4 = await process_input(
            input_data=url,
            model="gpt4",
            custom_prompt=custom_prompt,
            user_id=user_id,
            display_name=display_name,
            override_base_name=override_base_name
        )
        result_gpt4mini = await process_input(
            input_data=url,
            model="openai",  # GPT-4-mini option
            custom_prompt=custom_prompt,
            user_id=user_id,
            display_name=display_name,
            override_base_name=override_base_name
        )
        result_togetherai = await process_input(
            input_data=url,
            model="togetherai",
            custom_prompt=custom_prompt,
            user_id=user_id,
            display_name=display_name,
            override_base_name=override_base_name
        )
        return {"gpt4": result_gpt4, "gpt4mini": result_gpt4mini, "togetherai": result_togetherai}
    except Exception as e:
        logging.error(f"Error in /summarize/url: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize/upload")
async def summarize_upload(
    custom_prompt: str = Form(..., description="Your custom prompt for summarization"),
    file: UploadFile = File(..., description="The PDF file to summarize")
):
    """
    Summarize an uploaded PDF file.
    Generates three summaries: GPT-4, GPT-4-mini, and TogetherAI.
    Uses default guest values.
    """
    try:
        user_id = "guest_user"
        display_name = "Guest"
        file_content = await file.read()
        file_obj = BytesIO(file_content)
        file_obj.name = file.filename

        result_gpt4 = await process_input(
            input_data=file_obj,
            model="gpt4",
            custom_prompt=custom_prompt,
            user_id=user_id,
            display_name=display_name,
            override_base_name=file.filename
        )
        file_obj.seek(0)
        result_gpt4mini = await process_input(
            input_data=file_obj,
            model="openai",
            custom_prompt=custom_prompt,
            user_id=user_id,
            display_name=display_name,
            override_base_name=file.filename
        )
        file_obj.seek(0)
        result_togetherai = await process_input(
            input_data=file_obj,
            model="togetherai",
            custom_prompt=custom_prompt,
            user_id=user_id,
            display_name=display_name,
            override_base_name=file.filename
        )
        return {"gpt4": result_gpt4, "gpt4mini": result_gpt4mini, "togetherai": result_togetherai}
    except Exception as e:
        logging.error(f"Error in /summarize/upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Updated feedback endpoint: update the summary document in the user's summaries subcollection
@app.post("/feedback")
async def feedback(
    summary_id: str = Form(...),
    feedback: str = Form(...),  # "like" or "dislike"
    comment: str = Form(None)
):
    """
    Update the summary document with feedback (like/dislike and an optional comment).
    The summary document is updated in the "users/guest/summaries" subcollection.
    """
    try:
        # We assume the summary belongs to the guest user.
        summary_ref = db.collection("users").document("guest_user").collection("summaries").document(summary_id)
        update_data = {
            "feedback": feedback,
            "comment": comment,
            "feedback_timestamp": firestore.SERVER_TIMESTAMP
        }
        summary_ref.update(update_data)
        logging.info(f"Feedback updated for summary {summary_id}")
        return {"status": "success", "summary_id": summary_id}
    except Exception as e:
        logging.error(f"Error updating feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)