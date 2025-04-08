from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import logging
import os
import json
import base64
from datetime import timedelta

# Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, storage, firestore

# Your summarization logic
from summary import process_input

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Environment variables
firebase_credentials_b64 = os.getenv("FIREBASE_CREDENTIALS")
openai_api_key = os.getenv("OPENAI_API_KEY")
together_api_key = os.getenv("TOGETHER_API_KEY")

if not firebase_credentials_b64:
    raise ValueError("Firebase credentials are missing from environment variables!")

# Initialize Firebase
try:
    firebase_json = base64.b64decode(firebase_credentials_b64).decode("utf-8")
    firebase_json_data = json.loads(firebase_json)
    
    # Write to temp JSON file
    with open("/tmp/firebase.json", "w") as f:
        json.dump(firebase_json_data, f)
    
    if not firebase_admin._apps:
        cred = credentials.Certificate(firebase_json_data)
        firebase_admin.initialize_app(cred, {
            "storageBucket": f"{firebase_json_data.get('project_id')}.appspot.com"
        })
        logging.info("Firebase initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize Firebase: {str(e)}")
    raise

# Firestore client
db = firestore.client()

# FastAPI setup
app = FastAPI(
    title="Bill Summarization API",
    description="An API to generate bill summaries using GPT models.",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    return FileResponse("static/index.html")

@app.get("/list_files")
async def list_files():
    try:
        bucket = storage.bucket()
        blobs = bucket.list_blobs(prefix="users/guest_user/")
        
        if not blobs:
            return {"files": {}}
            
        files = {}
        for blob in blobs:
            if blob.name.endswith(".pdf"):
                file_name = blob.name.split("/")[-1]
                url = blob.generate_signed_url(
                    expiration=timedelta(minutes=15),
                    method="GET"
                )
                files[file_name] = url
        return {"files": files}
    except Exception as e:
        logging.error(f"Error listing files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize/url")
async def summarize_url(
    url: str = Form(...),
    custom_prompt: str = Form(...),
    override_base_name: str = Form("")
):
    try:
        if not url:
            raise HTTPException(status_code=400, detail="No URL provided")
        
        result = await process_input(
            input_data=url,
            model="gpt4",
            custom_prompt=custom_prompt or "Summarize the document.",
            user_id="guest_user",
            display_name="Guest",
            file_url=url if url.startswith("http") else None,
            override_base_name=override_base_name or ""
        )
        return result
    except Exception as e:
        logging.error(f"Error in summarize_url: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize/upload")
async def summarize_upload(
    file: UploadFile = File(...),
    custom_prompt: str = Form(...)
):
    try:
        result = await process_input(
            input_data=file,
            model="gpt4",
            custom_prompt=custom_prompt,
            user_id="guest_user",
            display_name="Guest"
        )
        return result
    except Exception as e:
        logging.error(f"Error in summarize_upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)