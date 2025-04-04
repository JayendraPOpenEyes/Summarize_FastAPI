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
        
        files = [blob.name.split("/")[-1] for blob in blobs if blob.name.endswith(".pdf")]
        return {"files": files}
    except Exception as e:
        logging.error(f"Error fetching files from Firebase Storage: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve files")

@app.post("/summarize/url")
async def summarize_url(request: Request):
    data = await request.json()
    url = data.get("url")
    if not url:
        raise HTTPException(status_code=400, detail="No URL provided")
    
    # Simulated summarization logic – replace with real implementation
    summary = f"This is a placeholder summary for the content at {url}."
    return {"summary": summary}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
