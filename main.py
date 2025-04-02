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

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Decode Firebase credentials from environment variable and write to a temporary JSON file
firebase_credentials_base64 = os.getenv("FIREBASE_CREDENTIALS")
if firebase_credentials_base64:
    firebase_credentials_json = base64.b64decode(firebase_credentials_base64).decode("utf-8")
    with open("/tmp/firebase.json", "w") as f:
        f.write(firebase_credentials_json)
else:
    raise ValueError("Firebase credentials are missing!")

# Initialize Firebase Admin SDK (ensure this only runs once)
if not firebase_admin._apps:
    cred = credentials.Certificate("/tmp/firebase.json")
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

@app.get("/", response_class=HTMLResponse)
async def read_index():
    return FileResponse("static/index.html")

# The rest of your FastAPI endpoints remain unchanged...

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
