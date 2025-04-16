import os
import json
import firebase_admin
from firebase_admin import credentials, firestore, storage
from dotenv import load_dotenv

load_dotenv()

if not firebase_admin._apps:
    firebase_cred_path = "firebase-adminsdk.json"
    if not os.path.exists(firebase_cred_path):
        raise FileNotFoundError("firebase-adminsdk.json not found in the project directory.")
    
    # Load Firebase credentials
    with open(firebase_cred_path, 'r') as f:
        creds = json.load(f)
    
    cred = credentials.Certificate(firebase_cred_path)
    bucket_name = f"{creds['project_id']}.appspot.com"
    
    # ✅ Initialize Firebase app (no need for custom database_id)
    firebase_admin.initialize_app(cred, {
        "storageBucket": bucket_name
    })

# ✅ Correct Firestore initialization (DO NOT add database_id unless using multi-database setup)
db = firestore.client(database_id="statside-summary")

# ✅ Firebase storage bucket
bucket = storage.bucket()

# Debug
print("Firestore connected:", db)
