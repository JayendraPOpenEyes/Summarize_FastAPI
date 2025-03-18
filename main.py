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

# Import your summary processing code
from summary import process_input

# Import Firebase storage (already initialized in summary.py)
from firebase_admin import storage

app = FastAPI(
    title="Bill Summarization API",
    description="An API to generate bill summaries using OpenAI and TogetherAI models.",
    version="1.0.0"
)

# Serve static files from the "static" directory.
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
    Uses default guest values.
    """
    try:
        user_id = "guest_user"
        display_name = "Guest"
        result_openai = await process_input(
            input_data=url,
            model="openai",
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
        return {"openai": result_openai, "togetherai": result_togetherai}
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
    Uses default guest values.
    """
    try:
        user_id = "guest_user"
        display_name = "Guest"
        file_content = await file.read()
        file_obj = BytesIO(file_content)
        file_obj.name = file.filename

        # Upload the file to Firebase Storage
        bucket = storage.bucket()
        blob = bucket.blob(f"users/{user_id}/{file.filename}")
        blob.upload_from_file(file_obj, rewind=True)  # Rewind the file pointer
        file_url = blob.generate_signed_url(expiration=timedelta(hours=1))  # Generate a signed URL

        result_openai = await process_input(
            input_data=file_obj,
            model="openai",
            custom_prompt=custom_prompt,
            user_id=user_id,
            display_name=display_name,
            override_base_name=file.filename,
            file_url=file_url #Passed the file url to firestore
        )
        file_obj.seek(0)
        result_togetherai = await process_input(
            input_data=file_obj,
            model="togetherai",
            custom_prompt=custom_prompt,
            user_id=user_id,
            display_name=display_name,
            override_base_name=file,
            file_url=file_url
        )
        return {"openai": result_openai, "togetherai": result_togetherai}
    except Exception as e:
        logging.error(f"Error in /summarize/upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
