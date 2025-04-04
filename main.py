from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import logging
import os
import json
import base64
from io import BytesIO
from datetime import timedelta
from dotenv import load_dotenv

# Firebase Admin SDK imports
import firebase_admin
from firebase_admin import credentials, storage, firestore

# Import your summarization functions and classes
from summary import process_input, TextProcessor

# Optional: load .env file for local testing
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def read_secret_file(path: str) -> str:
    """Helper function to read a secret from a mounted file."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read().strip()
    raise ValueError(f"Secret not found at: {path}")

# Read Firebase credentials from secret file
firebase_secret_path = "/secrets/firebase-credentials"
firebase_credentials_json = read_secret_file(firebase_secret_path)

# Write the Firebase credentials to a temporary file (if required by firebase_admin)
with open("/tmp/firebase.json", "w") as f:
    f.write(firebase_credentials_json)

# Initialize Firebase Admin SDK (ensure this only runs once)
if not firebase_admin._apps:
    cred = credentials.Certificate(json.loads(firebase_credentials_json))
    firebase_admin.initialize_app(cred, {
        "storageBucket": f"{os.getenv('GCP_PROJECT_ID')}.appspot.com"
    })

# Firestore DB client for any feedback or data updates.
db = firestore.client()

# Read API keys for OpenAI and Together AI from secret files
openai_secret_path = "/secrets/openai-key"
together_secret_path = "/secrets/together-key"
openai_api_key = read_secret_file(openai_secret_path)
together_api_key = read_secret_file(together_secret_path)

# Set your OpenAI API key (if using the openai client library)
import openai
openai.api_key = openai_api_key

# Set up the FastAPI app
app = FastAPI(
    title="Bill Summarization API",
    description="An API to generate bill summaries using GPT-4, GPT-4-mini, and TogetherAI models.",
    version="1.0.0"
)

# Mount static files from the "static" directory.
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    return FileResponse("static/index.html")

# Example endpoint for file upload and processing
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        # Process the file contents using your summarization function
        result = process_input(contents)
        return {"result": result}
    except Exception as e:
        logging.error("Error processing file: %s", e)
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
