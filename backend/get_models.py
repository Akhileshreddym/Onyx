import os
import requests
from dotenv import load_dotenv
load_dotenv(".env")
api_key = os.getenv("ELEVENLABS_API_KEY")
url = "https://api.elevenlabs.io/v1/models"
headers = {"xi-api-key": api_key}
response = requests.get(url, headers=headers)
models = response.json()
for m in models:
    print(f"Model ID: {m.get('model_id')}, Name: {m.get('name')}")
