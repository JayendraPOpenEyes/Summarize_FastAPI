from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import logging
import os
import json
import base64
from io import BytesIO
from datetime import timedelta

# Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, storage, firestore

# Your summarization logic
from summary import process_input, TextProcessor

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Environment variables
firebase_credentials_b64 = os.getenv("FIREBASE_CREDENTIALS")
openai_api_key = os.getenv("OPENAI_API_KEY")
together_api_key = os.getenv("TOGETHER_API_KEY")

if not firebase_credentials_b64:
    raise ValueError("Firebase credentials are missing from environment variables!")

# Decode Firebase credentials and write to temp JSON
firebase_json = base64.b64decode(firebase_credentials_b64).decode("utf-8")
with open("/tmp/firebase.json", "w") as f:
    f.write(firebase_json)

# Firebase Admin init
if not firebase_admin._apps:
    cred = credentials.Certificate(json.loads(firebase_json))
    firebase_admin.initialize_app(cred, {
        "storageBucket": 'project-astra-438804.appspot.com'
    })

# Firestore client
db = firestore.client()

# FastAPI setup
app = FastAPI(
    title="Bill Summarization API",
    description="An API to generate bill summaries using GPT-4, GPT-4-mini, and TogetherAI models.",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    return FileResponse("static/index.html")

from google.cloud import storage

@app.get("/list_files")
async def list_files():
    try:
        bucket = storage.Client().bucket("project-astra-438804.appspot.com")
        blobs = bucket.list_blobs(prefix="users/guest_user/")  # Adjust prefix as needed
        
        files = {}
        for blob in blobs:
            if blob.name.endswith(".pdf"):
                file_name = blob.name.split("/")[-1]
                # Generate a signed URL (valid for 15 minutes)
                url = blob.generate_signed_url(
                    expiration=timedelta(minutes=15),
                    method="GET"
                )
                files[file_name] = url
        return {"files": files}  # Return object with file names as keys and signed URLs as values
    except Exception as e:
        logging.error(f"Error fetching files from Firebase Storage: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve files")

@app.post("/summarize/url")
async def summarize_url(request: Request):
    data = await request.json()
    url = data.get("url")
    custom_prompt = data.get("custom_prompt")
    override_base_name = data.get("override_base_name", "")
    if not url:
        raise HTTPException(status_code=400, detail="No URL provided")
    
    # Process the input (URL or signed URL from chosen file)
    result = await process_input(
        input_data=url,
        model="gpt4",  # Default model, adjust as needed
        custom_prompt=custom_prompt or "Summarize the document.",
        user_id="guest_user",  # Adjust based on authentication
        display_name="Guest",
        file_url=url if url.startswith("http") else None,
        override_base_name=override_base_name or ""
    )
    return result

# Add a placeholder for /summarize/upload (implement as needed)
@app.post("/summarize/upload")
async def summarize_upload(file: UploadFile = File(...), custom_prompt: str = Form(...)):
    # Implement file upload and summarization logic here
    result = await process_input(
        input_data=file,
        model="gpt4",  # Default model, adjust as needed
        custom_prompt=custom_prompt,
        user_id="guest_user",  # Adjust based on authentication
        display_name="Guest"
    )
    return result

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)