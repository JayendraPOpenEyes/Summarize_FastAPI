import base64

with open("firebase-adminsdk.json", "rb") as f:
    encoded = base64.b64encode(f.read()).decode('utf-8')

print(encoded)  # Copy this to GitHub Secrets
