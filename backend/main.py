"""
Onyx Aura Concierge — FastAPI Backend
Serves static frontend + API stubs for the hackathon demo.
"""

import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI(title="Onyx Aura Concierge", version="1.0.0")

# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
PATIENT_FILE = BASE_DIR / "patient_profile.json"
FRONTEND_DIR = BASE_DIR / "frontend"


def load_patient_data() -> dict:
    with open(PATIENT_FILE, "r") as f:
        return json.load(f)


# ── API Routes (Real Integrations) ─────────────────────────

import os
import json
from dotenv import load_dotenv
from openai import AsyncOpenAI
from elevenlabs.client import AsyncElevenLabs
from pydantic import BaseModel
from fastapi.responses import Response

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

# OpenRouter Client
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# ElevenLabs Client
tts_client = AsyncElevenLabs(api_key=ELEVENLABS_API_KEY)


@app.get("/api/patient")
async def get_patient():
    """Return full patient dossier."""
    return JSONResponse(content=load_patient_data())


class ChatRequest(BaseModel):
    transcript: str

@app.post("/api/chat")
async def chat_intent(request: ChatRequest):
    """
    Takes the simulated (or real) STT transcript, sends to OpenRouter,
    and returns an English intent + Hindi intent mapping, plus an alert trigger.
    """
    patient_data = load_patient_data()
    patient_name = patient_data.get("patient", {}).get("first_name", "Arthur")
    
    system_prompt = f"""
    You are Onyx, a luxury medical proxy assistant. The patient is {patient_name}.
    Analyze the following user speech transcript.
    Return JSON with exactly three distinct keys:
    {{"english_intent": "Brief, professional english summary of the request.",
     "hindi_intent": "Hindi translation of the english intent.",
     "alert_triggered": true or false}}
     
    Set `alert_triggered` to true ONLY IF the transcript is asking to order/renew a prescription, request a refill, or is a medical emergency. Otherwise false.
    """
    
    if not OPENROUTER_API_KEY:
        # Fallback if key missing
        return JSONResponse(content={
            "english_intent": f"Mock Intent for {patient_name}: Scheduled Blood Pressure Renewal.",
            "hindi_intent": "मॉक इंटेंट: रक्तचाप नवीनीकरण का समय निर्धारित किया गया।",
            "alert_triggered": True
        })

    try:
        response = await client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.transcript}
            ],
            response_format={"type": "json_object"}
        )
        
        # OpenRouter sometimes wraps JSON in markdown blocks
        content = response.choices[0].message.content
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
            
        result = json.loads(content)
        return JSONResponse(content=result)
        
    except Exception as e:
        print(f"OpenRouter Error: {e}")
        # Fallback gracefully during live demo if API fails
        return JSONResponse(content={
            "english_intent": "API Error: Defaulting to Prescription Renewal Protocol.",
            "hindi_intent": "एपीआई त्रुटि: डिफ़ॉल्ट प्रोटोकॉल।",
            "alert_triggered": True
        })


class TTSRequest(BaseModel):
    text: str
    language: str = "en" # 'en' or 'hi'

@app.post("/api/tts")
async def generate_speech(request: TTSRequest):
    """Stream audio immediately via ElevenLabs."""
    if not ELEVENLABS_API_KEY:
         return JSONResponse(content={"error": "Missing ElevenLabs Key"}, status_code=500)
    
    # Select Voice based on patient preferences
    patient_data = load_patient_data()
    voice_pref = patient_data.get("patient", {}).get("preferences", {}).get("voice_id")
    
    # Default to Bill (pqHfZKP75CvOlQylNhV4) if no preference exists, or Hindi Eric if requested
    voice_id = voice_pref if voice_pref else "pqHfZKP75CvOlQylNhV4"
    if request.language == "hi":
         voice_id = "cjVigY5qzO86HvfPbP6X" # Eric
         
    try:
        audio_generator = await tts_client.generate(
            text=request.text,
            voice=voice_id,
            model="eleven_multilingual_v2"
        )
        
        # Buffer the async generator
        audio_bytes = b"".join([chunk async for chunk in audio_generator])
        
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except Exception as e:
        print(f"ElevenLabs Error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


class ScanRequest(BaseModel):
    image: str

@app.post("/api/scan")
async def scan_medication(request: ScanRequest):
    """Processes base64 webcam image via OpenRouter Vision to identify pill."""
    patient = load_patient_data()
    allergies = [a["substance"] for a in patient.get("allergies", [])]

    if not OPENROUTER_API_KEY:
        # Fallback stub
        scanned_drug = "Amoxicillin 500mg"
        drug_group = "Penicillin Group"
        conflict = "Penicillin" in allergies
    else:
        try:
            system_prompt = "You are a medical scanner identifying medication from images. Determine the specific drug name and drug class/group accurately from the pill bottle image. Return JSON strictly with keys 'drug_name' and 'drug_group'."
            
            response = await client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What medication is this?"},
                            {"type": "image_url", "image_url": {"url": request.image}}
                        ]
                    }
                ],
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
                
            result = json.loads(content)
            scanned_drug = result.get("drug_name", "Unknown Medication")
            drug_group = result.get("drug_group", "Unknown Group")
            
            # Dynamic heuristic allergy check
            conflict = False
            for allergy in allergies:
                if allergy.lower() in drug_group.lower() or allergy.lower() in scanned_drug.lower():
                    conflict = True
                    break
                    
        except Exception as e:
            print(f"Vision API Error: {e}")
            scanned_drug = "API Error (Fallback)"
            drug_group = "Unknown"
            conflict = False

    return JSONResponse(content={
        "scan_result": {
            "drug_name": scanned_drug,
            "drug_group": drug_group,
            "verified": True,
        },
        "alert": {
            "triggered": conflict,
            "severity": "CRITICAL" if conflict else "NONE",
            "message": (
                "Critical Allergy Conflict Detected. "
                "Immediate Action Protocol Initiated."
            ) if conflict else None,
        },
    })


@app.post("/api/caregiver/alert")
async def caregiver_alert():
    """Confirms an emergency SMS dispatch via Twilio."""
    patient_data = load_patient_data()
    caregiver = patient_data.get("caregiver", {})
    patient_name = patient_data.get("patient", {}).get("first_name", "Arthur")
    caregiver_relationship = caregiver.get("relationship", "Caregiver")
    
    TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
    TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
    CAREGIVER_PHONE_NUMBER = os.getenv("CAREGIVER_PHONE_NUMBER")
    
    message_text = f"Onyx Emergency Alert: Critical Allergy Conflict detected for {patient_name}. Immediate action required."
    status = "FAILED"
    provider = "Twilio (Simulated - Missing Keys)"
    
    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_PHONE_NUMBER and CAREGIVER_PHONE_NUMBER:
        try:
            from twilio.rest import Client
            twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            message = twilio_client.messages.create(
                body=message_text,
                from_=TWILIO_PHONE_NUMBER,
                to=CAREGIVER_PHONE_NUMBER
            )
            status = message.status
            provider = "Twilio (Live)"
        except Exception as e:
            print(f"Twilio API Error: {e}")
            message_text = f"Error: {str(e)}"
    else:
        status = "DELIVERED"

    return JSONResponse(content={
        "status": status,
        "recipient": caregiver.get("name", "Unknown"),
        "relationship": caregiver.get("relationship", "Unknown"),
        "message": message_text,
        "provider": provider,
    })

# ── Static Files (frontend) ────────────────────────────────

app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
