from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import logging
import os
from io import BytesIO
from datetime import timedelta
from dotenv import load_dotenv

# Firebase Admin SDK imports
import firebase_admin
from firebase_admin import credentials, storage, firestore

# Import your summarization functions and classes
from summary import process_input, TextProcessor

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize Firebase Admin SDK (ensure this only runs once)
if not firebase_admin._apps:
    cred = credentials.Certificate("path/to/your/serviceAccountKey.json")
    firebase_admin.initialize_app(cred, {
        "storageBucket": "your-project-id.appspot.com"
    })

# Firestore DB client for feedback updates.
db = firestore.client()

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
        bucket = storage.bucket()  # Using the default Firebase bucket
        prefix = "users/guest_user/"  # Folder for guest user files
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
    Uploads the file to Firebase Storage with duplicate naming logic and generates three summaries.
    """
    try:
        # Use a consistent user_id for storage and listing.
        user_id = "guest_user"
        display_name = "Guest"

        # Read the file content.
        file_content = await file.read()
        file_obj = BytesIO(file_content)
        file_obj.name = file.filename

        # Upload the file to Firebase Storage with duplicate name check.
        bucket = storage.bucket()  # Firebase bucket initialized via firebase_admin
        original_filename = file.filename  # e.g., "xyz.pdf"
        base_name, ext = os.path.splitext(original_filename)
        
        # Start with the original file name.
        new_filename = original_filename
        counter = 1
        blob_path = f"users/{user_id}/{new_filename}"
        
        # Loop until a file with the new_filename does not exist.
        while bucket.blob(blob_path).exists():
            new_filename = f"{base_name}({counter}){ext}"
            blob_path = f"users/{user_id}/{new_filename}"
            counter += 1

        blob = bucket.blob(blob_path)
        blob.upload_from_string(file_content, content_type=file.content_type)
        # Optionally, generate a signed URL for accessing the uploaded file.
        file_url = blob.generate_signed_url(expiration=timedelta(hours=1))

        # Process the file for summarization.
        extraction_processor = TextProcessor("openai")
        base_name_used = file_obj.name  # original filename for summarization processing
        _, ext = os.path.splitext(base_name_used)
        ext = ext.lower()
        if ext == ".pdf":
            extraction_result = extraction_processor.process_uploaded_pdf(file_obj, base_name=base_name_used)
        elif ext in [".htm", ".html"]:
            extraction_result = extraction_processor.process_uploaded_html(file_obj, base_name=base_name_used)
        else:
            return {"error": "Unsupported file type. Please upload a PDF or HTML file."}

        if extraction_result.get("error"):
            return {"error": extraction_result["error"]}

        clean_text = extraction_processor.preprocess_text(extraction_result["text"])

        # Generate summaries using different processors.
        processor_gpt4 = TextProcessor("gpt4")
        result_gpt4 = processor_gpt4.generate_summary(clean_text, base_name_used, custom_prompt, user_id, display_name)

        processor_gpt4mini = TextProcessor("openai")
        result_gpt4mini = processor_gpt4mini.generate_summary(clean_text, base_name_used, custom_prompt, user_id, display_name)

        processor_togetherai = TextProcessor("togetherai")
        result_togetherai = processor_togetherai.generate_summary(clean_text, base_name_used, custom_prompt, user_id, display_name)

        return {
            "status": "success",
            "file_url": file_url,
            "uploaded_filename": new_filename,
            "gpt4": result_gpt4,
            "gpt4mini": result_gpt4mini,
            "togetherai": result_togetherai
        }

    except Exception as e:
        logging.error(f"Error in /summarize/upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def feedback(
    summary_id: str = Form(...),
    feedback: str = Form(...),  # "like" or "dislike"
    comment: str = Form(None)
):
    """
    Update the summary document with feedback (like/dislike and an optional comment).
    The summary document is updated in the "users/guest_user/summaries" subcollection.
    """
    try:
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
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
