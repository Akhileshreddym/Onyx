"""
Onyx Concierge — FastAPI Backend
Serves static frontend + API stubs for the hackathon demo.
"""
from __future__ import annotations

import json
from typing import Union, Optional, List
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response

app = FastAPI(title="Onyx Concierge", version="1.0.0")

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


def save_patient_data(data: dict) -> None:
    with open(PATIENT_FILE, "w") as f:
        json.dump(data, f, indent=2)


def get_refill_due_medications() -> list:
    """Return list of medications due for refill (today >= next_refill - reminder_days_before)."""
    patient = load_patient_data()
    meds = patient.get("current_medications", [])
    today = date.today()
    due = []
    for m in meds:
        name = m.get("name")
        last_fill = m.get("last_fill_date")
        days_supply = m.get("days_supply")
        reminder_days = m.get("reminder_days_before", 3)
        if not name or last_fill is None or days_supply is None:
            continue
        try:
            if isinstance(last_fill, str):
                last = datetime.strptime(last_fill.strip()[:10], "%Y-%m-%d").date()
            else:
                continue
            days_int = int(days_supply)
            reminder_int = int(reminder_days)
        except (ValueError, TypeError):
            continue
        run_out = last + timedelta(days=days_int)
        remind_from = run_out - timedelta(days=reminder_int)
        if today >= remind_from:
            due.append({**m, "next_refill_date": run_out.isoformat()})
    return due


def get_refill_upcoming_medications(days_ahead: int = 14) -> list:
    """Return medications whose reminder date is in the next days_ahead days (not yet due)."""
    due_names = {m.get("name") for m in get_refill_due_medications() if m.get("name")}
    patient = load_patient_data()
    meds = patient.get("current_medications", [])
    today = date.today()
    window_end = today + timedelta(days=days_ahead)
    upcoming = []
    for m in meds:
        name = m.get("name")
        if name in due_names:
            continue
        last_fill = m.get("last_fill_date")
        days_supply = m.get("days_supply")
        reminder_days = m.get("reminder_days_before", 3)
        if not name or last_fill is None or days_supply is None:
            continue
        try:
            if isinstance(last_fill, str):
                last = datetime.strptime(last_fill.strip()[:10], "%Y-%m-%d").date()
            else:
                continue
            days_int = int(days_supply)
            reminder_int = int(reminder_days)
        except (ValueError, TypeError):
            continue
        run_out = last + timedelta(days=days_int)
        remind_from = run_out - timedelta(days=reminder_int)
        if today < remind_from and remind_from <= window_end:
            days_until = (remind_from - today).days
            upcoming.append({
                **m,
                "next_refill_date": run_out.isoformat(),
                "remind_from": remind_from.isoformat(),
                "days_until_reminder": days_until,
            })
    return upcoming


def _refill_reminder_job() -> None:
    """Daily job: mark refill reminder state for due meds so we can avoid spamming."""
    try:
        due = get_refill_due_medications()
        if not due:
            return
        patient = load_patient_data()
        reminders = patient.get("refill_reminders", {})
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        for m in due:
            name = m.get("name")
            if name:
                reminders[name] = now
        patient["refill_reminders"] = reminders
        save_patient_data(patient)
    except Exception as e:
        print(f"Refill reminder job error: {e}")


_refill_scheduler = None
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    _refill_scheduler = BackgroundScheduler()
    _refill_scheduler.add_job(_refill_reminder_job, "cron", hour=9, minute=0)  # daily at 9:00
except ImportError:
    pass


@app.on_event("startup")
def _start_refill_scheduler():
    if _refill_scheduler is not None and not _refill_scheduler.running:
        _refill_scheduler.start()


@app.on_event("shutdown")
def _stop_refill_scheduler():
    if _refill_scheduler is not None:
        _refill_scheduler.shutdown()


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

@app.get("/onboarding")
async def serve_onboarding():
    return FileResponse(FRONTEND_DIR / "onboarding.html")

from fastapi import FastAPI, Request, UploadFile, File, Form
import json
import os
import base64
import fitz  # PyMuPDF
from PIL import Image
import io

@app.post("/api/process_report")
async def process_report(file: UploadFile = File(...)):
    """Process an uploaded PDF or Image via Gemini 1.5 Pro to extract patient profile data."""
    print(f"File received: {file.filename}")
    
    content_type = file.content_type
    file_bytes = await file.read()
    
    messages = [
        {
            "role": "system",
            "content": "You are Onyx, an advanced medical AI assistant. Analyze the provided medical or demographic document and extract the patient's full details into a precise JSON structure. Match the required JSON schema EXACTLY. Output ONLY raw JSON, with no markdown code blocks, no backticks, and no explanations."
        }
    ]
    
    # Process based on file type
    if content_type == "application/pdf":
        text_content = ""
        try:
            # Wrap bytes in a BytesIO object for PyMuPDF
            pdf_stream = io.BytesIO(file_bytes)
            doc = fitz.open(stream=pdf_stream, filetype="pdf")
            for page in doc:
                text_content += page.get_text() + "\n"
        except Exception as e:
            print(f"PDF Parse Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"error": f"Failed to parse PDF document: {str(e)}"})
            
        messages.append({
            "role": "user",
            "content": f"Extract patient profile from this text:\n\n{text_content}\n\nSchema Requirements:\n{{\"patient\":{{\"id\":\"...\",\"first_name\":\"...\",\"last_name\":\"...\",\"date_of_birth\":\"MM/DD/YYYY\",\"age\":0,\"gender\":\"...\",\"location\":{{\"city\":\"...\",\"state\":\"...\",\"zip\":\"...\"}},\"primary_physician\":\"...\",\"insurance\":\"...\"}},\"allergies\":[{{\"substance\":\"...\",\"severity\":\"...\",\"reaction\":\"...\",\"priority\":\"HIGH/LOW\",\"verified\":true}}],\"current_medications\":[{{\"name\":\"...\",\"dosage\":\"...\",\"frequency\":\"...\",\"purpose\":\"...\",\"days_supply\":30,\"last_fill_date\":\"YYYY-MM-DD\",\"reminder_days_before\":3}}],\"caregiver\":{{\"relationship\":\"...\",\"name\":\"...\",\"phone\":\"...\",\"connection_status\":\"Active\",\"sms_alerts_enabled\":true,\"notification_provider\":\"Twilio\"}},\"emergency_log\":[]}}"
        })
    elif content_type.startswith("image/"):
        base64_img = base64.b64encode(file_bytes).decode('utf-8')
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Extract patient profile from this image. Schema Requirements:\n{{\"patient\":{{\"id\":\"...\",\"first_name\":\"...\",\"last_name\":\"...\",\"date_of_birth\":\"MM/DD/YYYY\",\"age\":0,\"gender\":\"...\",\"location\":{{\"city\":\"...\",\"state\":\"...\",\"zip\":\"...\"}},\"primary_physician\":\"...\",\"insurance\":\"...\"}},\"allergies\":[{{\"substance\":\"...\",\"severity\":\"...\",\"reaction\":\"...\",\"priority\":\"HIGH/LOW\",\"verified\":true}}],\"current_medications\":[{{\"name\":\"...\",\"dosage\":\"...\",\"frequency\":\"...\",\"purpose\":\"...\",\"days_supply\":30,\"last_fill_date\":\"YYYY-MM-DD\",\"reminder_days_before\":3}}],\"caregiver\":{{\"relationship\":\"...\",\"name\":\"...\",\"phone\":\"...\",\"connection_status\":\"Active\",\"sms_alerts_enabled\":true,\"notification_provider\":\"Twilio\"}},\"emergency_log\":[]}}"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{content_type};base64,{base64_img}"
                    }
                }
            ]
        })
    else:
        return JSONResponse(status_code=400, content={"error": "Unsupported file type. Please upload a PDF or Image."})
        
    try:
        response = await client.chat.completions.create(
            model="google/gemini-2.0-flash-001",
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        raw_output = response.choices[0].message.content
        print(f"Gemini Raw Output: {raw_output}")
        
        # Clean up any potential markdown formatting from the response
        cleaned_output = raw_output.strip()
        if cleaned_output.startswith("```json"):
            cleaned_output = cleaned_output[7:]
        if cleaned_output.startswith("```"):
            cleaned_output = cleaned_output[3:]
        if cleaned_output.endswith("```"):
            cleaned_output = cleaned_output[:-3]
        cleaned_output = cleaned_output.strip()
            
        profile_data = json.loads(cleaned_output)
        # Preserve scan_history and refill_reminders if present in existing file
        try:
            existing = load_patient_data()
            if "scan_history" in existing:
                profile_data["scan_history"] = existing["scan_history"]
            if "refill_reminders" in existing:
                profile_data["refill_reminders"] = existing["refill_reminders"]
        except Exception:
            pass
        # Save to patient_profile.json
        save_patient_data(profile_data)
            
        return JSONResponse(content={"status": "success", "profile": profile_data})
        
    except Exception as e:
        print(f"Gemini extraction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"Failed to extract profile: {str(e)}"})


class ChatRequest(BaseModel):
    transcript: str

@app.post("/api/chat")
async def chat_intent(request: ChatRequest):
    """
    Takes the simulated (or real) STT transcript, sends to OpenRouter,
    and returns an English intent plus an alert trigger.
    """
    patient_data = load_patient_data()
    patient_name = patient_data.get("patient", {}).get("first_name", "Arthur")
    
    system_prompt = f"""
    You are Onyx, a luxury medical proxy assistant. The patient is {patient_name}.
    Analyze the following user speech transcript.
    Return JSON with exactly three keys:
    {{"english_intent": "Brief, professional summary of the request.",
     "response_text": "A natural, conversational spoken reply to the user. If no action is needed, provide a helpful answer. If a pharmacy call is needed, confirm you are placing the call.",
     "alert_triggered": true or false}}
     
    Set `alert_triggered` to true ONLY IF the transcript is asking to order/renew a prescription, request a refill, or is a medical emergency. Otherwise false.
    
    Examples:
    - User: "How often can I take Tylenol?" -> alert_triggered: false, response_text: "You can take Tylenol every 4 to 6 hours as needed, but don't exceed 3,000 milligrams in 24 hours. Please consult your doctor if you need it frequently."
    - User: "Please order my blood pressure medication" -> alert_triggered: true, response_text: "Of course, {patient_name}. I'm placing a call to your pharmacy right now to order your blood pressure medication."
    """
    
    if not OPENROUTER_API_KEY:
        # Fallback if key missing
        return JSONResponse(content={
            "english_intent": f"Mock Intent for {patient_name}: Scheduled Blood Pressure Renewal.",
            "response_text": f"I'm placing a call to your pharmacy now to order your prescription, {patient_name}.",
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
        
        # Ensure response_text exists (fallback if model omits it)
        if "response_text" not in result:
            result["response_text"] = result.get("english_intent", "I'm here to help.")
        
        return JSONResponse(content=result)
        
    except Exception as e:
        print(f"OpenRouter Error: {e}")
        # Fallback gracefully during live demo if API fails
        return JSONResponse(content={
            "english_intent": "API Error: Defaulting to Prescription Renewal Protocol.",
            "response_text": f"I'm sorry, I encountered an error. Let me try to help you, {patient_name}.",
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
    
    # Default to Bill (pqHfZKP75CvOlQylNhV4) if no preference exists
    voice_id = voice_pref if voice_pref else "pqHfZKP75CvOlQylNhV4"
    
    try:
        audio_generator = tts_client.text_to_speech.convert(
            text=request.text,
            voice_id=voice_id,
            model_id="eleven_v3"
        )
        
        # Buffer the async generator
        audio_bytes = b"".join([chunk async for chunk in audio_generator])
        
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except Exception as e:
        print(f"ElevenLabs Error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ── CONVERSATIONAL v3 ENDPOINT (for two-way calls) ─────────────

from fastapi.responses import StreamingResponse
import asyncio

class ConversationState:
    """Manages conversation context across turns."""
    def __init__(self):
        self.messages = []
        self.detected_language = "en"
        self.patient_name = "Arthur"
        self.intent = None
        
    def add_user_message(self, text: str, language: str):
        self.messages.append({"role": "user", "content": text, "language": language})
        self.detected_language = language
        
    def add_assistant_message(self, text: str):
        self.messages.append({"role": "assistant", "content": text})

# Global audio cache for serving files directly
audio_cache = {}
audio_cache_counter = 0

# Create audio cache dir
AUDIO_CACHE_DIR = BASE_DIR / "audio_cache"
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

# Configurable base URL for audio serving (change this to your public URL)
AUDIO_BASE_URL = os.getenv("AUDIO_BASE_URL", "http://127.0.0.1:8000")

# Conversation state tracking
conversation_states = {}

def upload_bytes_to_cdn(audio_bytes: bytes, filename: str) -> str:
    """Uploads audio bytes directly to uguu.se CDN to bypass Ngrok requirements constraints for Twilio Play"""
    import requests
    try:
        print(f"  ⬆️ Uploading {filename} to CDN...")
        response = requests.post("https://uguu.se/upload", files={"files[]": (filename, audio_bytes, "audio/mpeg")}, timeout=15)
        data = response.json()
        url = data["files"][0]["url"]
        print(f"  ✓ Uploaded to CDN: {url}")
        return url
    except Exception as e:
        print(f"  ✗ CDN Upload Failed: {e}")
        return ""

@app.post("/api/convaI/message")
async def conversation_message(call_sid: str, user_speech: str, detected_lang: str = "en"):
    """
    Processes one turn of conversation.
    Called by Twilio during the call with new user input.
    """
    patient_data = load_patient_data()
    patient_name = patient_data.get("patient", {}).get("first_name", "Arthur")
    voice_pref = patient_data.get("patient", {}).get("preferences", {}).get("voice_id", "pqHfZKP75CvOlQylNhV4")
    
    # Initialize conversation state if needed
    if call_sid not in conversation_states:
        conversation_states[call_sid] = ConversationState()
        conversation_states[call_sid].patient_name = patient_name
    
    state = conversation_states[call_sid]
    
    # ElevenLabs v3 handles all 70+ languages automatically - use one consistent voice
    # No per-language voice switching needed; the model infers accent from the text
    voice_id = voice_pref if voice_pref else "EXAVITQu4vr4xnSDxMaL"  # Bill
    
    print(f"\n🎤 CONVERSATION TURN")
    print(f"  Call SID: {call_sid}")
    print(f"  User Speech: {user_speech}")
    print(f"  Language: {detected_lang}")
    
    state.add_user_message(user_speech, detected_lang)
    
    try:
        # 1. Use Gemini to understand medical intent and generate response
        system_message = f"""You are Onyx Concierge, a multilingual medical proxy assistant for {patient_name}. 
You understand medical requests in multiple languages and provide helpful, professional responses.
Current conversation language: {detected_lang}

IMPORTANT: Always respond in {detected_lang} (the patient's language).
Keep responses concise and focused on the medical/prescription need.
If discussing pharmacy changes or prescription needs, be clear and specific."""
        
        messages_for_gemini = [
            {"role": "system", "content": system_message},
        ]
        
        # Add conversation history
        for msg in state.messages:
            if msg["role"] == "user":
                messages_for_gemini.append({
                    "role": "user",
                    "content": msg["content"]
                })
            else:
                messages_for_gemini.append({
                    "role": "assistant",
                    "content": msg["content"]
                })
        
        response = await client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=messages_for_gemini,
            temperature=0.7,
            max_tokens=150
        )
        
        ai_response = response.choices[0].message.content
        state.add_assistant_message(ai_response)
        
        print(f"  ✓ Gemini Response: {ai_response}")
        
        # 2. Convert response to speech using ElevenLabs v3 in the user's language
        audio_generator = tts_client.text_to_speech.convert(
            text=ai_response,
            voice_id=voice_id,
            model_id="eleven_v3"
        )
        
        audio_bytes = b"".join([chunk async for chunk in audio_generator])
        print(f"  ✓ TTS Generated: {len(audio_bytes)} bytes")
        
        # 3. Check if we need to trigger pharmacy call (detect keywords)
        trigger_pharmacy = any(keyword in user_speech.lower() for keyword in 
                              ["pharmacy", "refill", "prescription", "medicine", "medication"])
        
        return JSONResponse(content={
            "status": "success",
            "response_text": ai_response,
            "language": detected_lang,
            "audio_size": len(audio_bytes),
            "trigger_pharmacy_call": trigger_pharmacy,
            "conversation_id": call_sid
        })
        
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        return JSONResponse(content={
            "status": "error",
            "error": str(e),
            "conversation_id": call_sid
        }, status_code=500)


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
        drug_description = "An antibiotic used to treat bacterial infections such as ear infections, pneumonia, and urinary tract infections."
        conflict = "Penicillin" in allergies
    else:
        try:
            allergies_str = ", ".join(allergies) if allergies else "None known"
            
            system_prompt = (
                "You are a medical scanner identifying medication from images. "
                "Determine the specific drug name, drug class/group, and a brief 1-sentence plain-language "
                "description of what the medication is used to treat.\n"
                f"CRITICAL: The patient has the following known allergies: {allergies_str}. "
                "Evaluate logically and medically if the scanned medication triggers any of these known allergies.\n"
                "Return JSON strictly with EXACTLY four keys: 'drug_name', 'drug_group', 'drug_description', and 'allergy_conflict' (boolean)."
            )

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
            drug_description = result.get("drug_description", "")
            
            # Rely on LLM's direct medical evaluation of the conflict
            conflict = result.get("allergy_conflict", False)

        except Exception as e:
            print(f"Vision API Error: {e}")
            scanned_drug = "API Error (Fallback)"
            drug_group = "Unknown"
            drug_description = ""
            conflict = False

    # Persist to scan history (cap at 100)
    try:
        patient_data = load_patient_data()
        scan_history = patient_data.get("scan_history", [])
        entry = {
            "id": str(uuid.uuid4()),
            "drug_name": scanned_drug,
            "drug_group": drug_group,
            "drug_description": drug_description,
            "scanned_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "bookmarked": False,
            "allergy_conflict": conflict,
        }
        scan_history.insert(0, entry)
        patient_data["scan_history"] = scan_history[:100]
        save_patient_data(patient_data)
    except Exception as e:
        print(f"Scan history save error: {e}")

    return JSONResponse(content={
        "scan_result": {
            "drug_name": scanned_drug,
            "drug_group": drug_group,
            "drug_description": drug_description,
            "verified": True,
        },
        "alert": {
            "triggered": conflict,
            "severity": "CRITICAL" if conflict else "NONE",
            "message": (
                "CRITICAL ALLERGEN DETECTED. DO NOT CONSUME. AVOID AT ALL COSTS."
            ) if conflict else None,
        },
    })


@app.get("/api/scan/history")
async def get_scan_history(bookmarked_only: bool = False):
    """Return scan history, optionally filtered to bookmarked only. Sorted by scanned_at descending."""
    patient = load_patient_data()
    history = patient.get("scan_history", [])
    if bookmarked_only:
        history = [h for h in history if h.get("bookmarked")]
    return JSONResponse(content={"scan_history": history})


class BookmarkUpdate(BaseModel):
    bookmarked: bool


@app.patch("/api/scan/history/{entry_id}")
async def update_scan_bookmark(entry_id: str, body: BookmarkUpdate):
    """Toggle bookmark for a scan history entry."""
    patient = load_patient_data()
    history = patient.get("scan_history", [])
    for entry in history:
        if entry.get("id") == entry_id:
            entry["bookmarked"] = body.bookmarked
            save_patient_data(patient)
            return JSONResponse(content={"ok": True, "id": entry_id, "bookmarked": body.bookmarked})
    return JSONResponse(content={"error": "Entry not found"}, status_code=404)


@app.get("/api/refill/status")
async def refill_status():
    """Return medications due for refill for the UI banner."""
    due = get_refill_due_medications()
    names = [m.get("name") for m in due if m.get("name")]
    message = f"Refill due for: {', '.join(names)}. Get refills on time — want us to contact your pharmacist?" if names else None
    return JSONResponse(content={"due_medications": due, "message": message})


class RefillPharmacyRequest(BaseModel):
    medication_names: Optional[Union[List[str], str]] = "all"  # list of names or "all" for all due


@app.post("/api/refill/request-pharmacy")
async def refill_request_pharmacy(body: RefillPharmacyRequest, http_request: Request):
    """Request pharmacy call for refill of due medications. Reuses caregiver alert flow."""
    patient = load_patient_data()
    patient_name = patient.get("patient", {}).get("first_name", "Patient")
    due = get_refill_due_medications()
    if body.medication_names == "all" or body.medication_names is None:
        med_list = due
    else:
        # Single str must be wrapped so set() doesn't iterate over characters
        names_iter = body.medication_names if isinstance(body.medication_names, list) else [body.medication_names]
        names_set = set(names_iter)
        med_list = [m for m in due if m.get("name") in names_set]
    if not med_list:
        return JSONResponse(content={"error": "No due medications to request", "status": "skipped"}, status_code=400)
    intent = "Routine refill for " + ", ".join(m["name"] for m in med_list) + " to ensure the patient gets their refills on time."
    call_request = PharmacyCallRequest(user_request=intent, patient_name=patient_name, intent=intent)
    return await caregiver_alert(call_request, http_request)


@app.get("/api/notifications")
async def get_notifications():
    """Unified notifications for the header bell: refill reminders + recent allergy-scan alerts."""
    patient = load_patient_data()
    first_name = (patient.get("patient") or {}).get("first_name", "Patient")
    notifications = []
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    # Refill due: one grouped notification
    due = get_refill_due_medications()
    if due:
        names = [m.get("name") for m in due if m.get("name")]
        if names:
            names_str = ", ".join(names)
            notifications.append({
                "id": "refill-" + now_iso,
                "type": "refill",
                "title": "Refill due",
                "message": f"{first_name}, refill due for {names_str}. Get refills on time — want us to contact your pharmacist?",
                "action": "contact_pharmacist",
                "created_at": now_iso,
            })

    # Refill upcoming: one notification per med (reminder in next 14 days)
    for m in get_refill_upcoming_medications(14):
        name = m.get("name")
        next_refill = m.get("next_refill_date", "")
        days_until = m.get("days_until_reminder")
        if not name:
            continue
        msg = f"{first_name}, {name} refill in {days_until} days (by {next_refill[:10]})." if days_until is not None and next_refill else f"{first_name}, {name} refill coming up."
        notifications.append({
            "id": f"refill-upcoming-{name}-{now_iso}",
            "type": "refill_upcoming",
            "title": "Refill coming up",
            "message": msg,
            "action": "view_refill",
            "created_at": now_iso,
            "next_refill_date": next_refill,
            "days_until": days_until,
        })

    # Allergy scan: recent scan_history entries with allergy_conflict (e.g. last 7 days or last 10)
    history = patient.get("scan_history", [])
    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat().replace("+00:00", "Z")
    allergy_entries = [h for h in history if h.get("allergy_conflict") and (h.get("scanned_at") or "") >= cutoff][:10]
    for i, entry in enumerate(allergy_entries):
        drug = entry.get("drug_name") or "A medication"
        scanned_at = entry.get("scanned_at") or now_iso
        notifications.append({
            "id": entry.get("id") or f"allergy-scan-{i}-{scanned_at}",
            "type": "allergy_scan",
            "title": "Allergy alert",
            "message": f"{drug} you scanned may conflict with your allergies.",
            "action": "view_scan_history",
            "created_at": scanned_at,
        })

    return JSONResponse(content={"notifications": notifications})


# ── TWILIO CONVERSATION HANDLER  ─────────────────────────

from twilio.twiml.voice_response import VoiceResponse, Gather

@app.post("/api/twilio/conversation")
async def twilio_conversation(request: Request):
    """
    TwiML endpoint for Twilio to call during conversation.
    Handles speech input, processes with Gemini, and responds in patient's language.
    """
    import json
    
    # Check form data from Twilio
    form_data = await request.form()
    call_sid = str(form_data.get("CallSid", "default-call"))
    speech_result = str(form_data.get("SpeechResult", ""))
    
    patient_data = load_patient_data()
    patient_name = patient_data.get("patient", {}).get("first_name", "Arthur")
    voice_pref = patient_data.get("patient", {}).get("preferences", {}).get("voice_id", "pqHfZKP75CvOlQylNhV4")
    
    print(f"\n📞 TWILIO CONVERSATION HANDLER")
    print(f"  Call SID: {call_sid}")
    print(f"  Speech: {speech_result}")
    
    try:
        # Initialize conversation if needed
        if call_sid not in conversation_states:
            conversation_states[call_sid] = ConversationState()
            conversation_states[call_sid].patient_name = patient_name
        
        state = conversation_states[call_sid]
        
        # ElevenLabs v3 handles all 70+ languages natively from the text itself
        # Just default to English for the AI response prompt
        detected_lang = "en"
        if speech_result:
            if any(word in speech_result.lower() for word in ["por favor", "gracias", "necesito"]):
                detected_lang = "es"
            elif any(word in speech_result.lower() for word in ["s'il vous plaît", "merci", "j'ai besoin"]):
                detected_lang = "fr"
        
        print(f"  Detected Language: {detected_lang}")
        
        # If first turn, send greeting
        if not speech_result:
            ai_response = f"Hello {patient_name}, I'm Onyx Concierge. How can I help you with your prescriptions today?"
        else:
            # Process with Gemini
            state.add_user_message(speech_result, detected_lang)
            
            system_message = f"""You are Onyx Concierge, a professional medical proxy for {patient_name}.
You must respond ONLY in {detected_lang} language.
Keep responses short (1-2 sentences) for phone calls.
Be helpful, professional, and focused on the medical request."""
            
            messages_for_gemini = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": speech_result}
            ]
            
            response = await client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=messages_for_gemini,
                temperature=0.7,
                max_tokens=100
            )
            
            ai_response = response.choices[0].message.content
            state.add_assistant_message(ai_response)
        
        print(f"  ✓ Response: {ai_response}")
        
        # ElevenLabs v3 handles all 70+ languages from the text — one voice works for all
        voice_id = voice_pref if voice_pref else "EXAVITQu4vr4xnSDxMaL"  # Bill
        
        audio_generator = tts_client.text_to_speech.convert(
            text=ai_response,
            voice_id=voice_id,
            model_id="eleven_v3"
        )
        
        audio_bytes = b"".join([chunk async for chunk in audio_generator])
        
        # Save audio for Twilio to fetch
        global audio_cache_counter
        audio_id = f"conversation_{audio_cache_counter}"
        audio_cache_counter += 1
        
        
        # Build URLs prioritizing BASE_URL env variable so Twilio webhook can parse public URLs
        base_url = os.getenv("BASE_URL")
        if not base_url:
            base_url = str(request.base_url).rstrip("/")
        else:
            base_url = base_url.rstrip("/")
            
        public_mp3_url = upload_bytes_to_cdn(audio_bytes, f"{audio_id}.mp3")
        if not public_mp3_url:
            public_mp3_url = f"{base_url}/api/audio/{audio_id}"
        
        # Build TwiML response
        response_obj = VoiceResponse()
        
        # Play the generated speech
        response_obj.play(public_mp3_url)
        
        # Gather next input (speech-to-text)
        gather = Gather(
            input="speech",
            timeout="3",
            speechTimeout="auto",
            language="en-US",
            action=f"{base_url}/api/twilio/conversation",
            method="POST"
        )
        gather.say(f"Please continue. I'm listening.", voice="alice")
        response_obj.append(gather)
        
        # Fallback after timeout
        response_obj.redirect(f"{base_url}/api/twilio/conversation")
        
        return Response(content=str(response_obj), media_type="application/xml")
        
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        response_obj = VoiceResponse()
        response_obj.say(f"Sorry, there was an error. Goodbye.")
        response_obj.hangup()
        return Response(content=str(response_obj), media_type="application/xml")


class PharmacyCallRequest(BaseModel):
    user_request: str
    patient_name: str = "Arthur"
    intent: str = ""

@app.post("/api/caregiver/alert")
async def caregiver_alert(request: PharmacyCallRequest, http_request: Request):
    """
    Makes a call to the pharmacy (CAREGIVER_PHONE_NUMBER) to execute the prescription action.
    Speaks TO the pharmacy on behalf of the patient, does NOT ask patient questions.
    """
    patient_data = load_patient_data()
    patient_full_name = patient_data.get("patient", {}).get("first_name", "Arthur")
    voice_pref = patient_data.get("patient", {}).get("preferences", {}).get("voice_id", "pqHfZKP75CvOlQylNhV4")
    
    TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
    TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
    CAREGIVER_PHONE_NUMBER = os.getenv("CAREGIVER_PHONE_NUMBER")  # This is pharmacy in demo
    
    # Create professional message to pharmacy
    user_request = request.user_request
    intent = request.intent
    
    patient_id = patient_data.get("patient", {}).get("id", "Unknown ID")
    patient_dob = patient_data.get("patient", {}).get("date_of_birth", "Unknown DOB")
    
    # Format the medications list string for audio TTS
    medications_list = ""
    current_meds = patient_data.get("current_medications", [])
    if current_meds:
        medications_list = ", ".join([f"{med.get('name', 'Unknown')} at {med.get('dosage', 'Unknown dosage')}" for med in current_meds])
    else:
        medications_list = "no currently listed medications"
        
    pharmacy_message = f"""Hello, this is Onyx Concierge, an automated medical proxy calling on behalf of {patient_full_name}. 
Patient Medical Record Number: {patient_id}. 
Date of Birth: {patient_dob}. 

The patient is requesting the following action: {intent}. 

For reference, their current active prescriptions include: {medications_list}. 

Please press 1 or say 'yes' to confirm you can process this request."""
    
    status = "FAILED"
    provider = "Twilio Pharmacy Call (ElevenLabs v3)"
    debug_info = ""
    
    print(f"\n📞 PHARMACY CALL INITIATED")
    print(f"  Patient: {patient_full_name}")
    print(f"  Request: {user_request}")
    print(f"  Intent: {intent}")
    print(f"  Calling Pharmacy: {CAREGIVER_PHONE_NUMBER}")
    print(f"  From: {TWILIO_PHONE_NUMBER}")
    
    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_PHONE_NUMBER and CAREGIVER_PHONE_NUMBER:
        try:
            import tempfile
            import requests
            
            debug_info = "Synthesizing message for pharmacy..."
            print(f"  ✓ {debug_info}")
            
            # 1. Synthesize ElevenLabs Turbo v2.5 (latest, emotional, multilingual)
            audio_generator = tts_client.text_to_speech.convert(
                text=pharmacy_message,
                voice_id=voice_pref,
                model_id="eleven_turbo_v2_5",
                optimize_streaming_latency="3"
            )
            audio_bytes = b"".join([chunk async for chunk in audio_generator])
            debug_info = f"TTS complete. Audio size: {len(audio_bytes)} bytes"
            print(f"  ✓ {debug_info}")
            
            # 2. Cache audio to file
            global audio_cache_counter
            audio_id = f"pharmacy_{audio_cache_counter}"
            audio_cache_counter += 1
            
            # Save to file
            audio_file_path = os.path.join(AUDIO_CACHE_DIR, f"{audio_id}.mp3")
            with open(audio_file_path, "wb") as f:
                f.write(audio_bytes)
            
            # Use the request base url since that is what the frontend is using
            base_url = str(http_request.base_url).rstrip("/")
            
            # Force-reload .env so we always get the latest tunnel URL (it may have changed after boot)
            load_dotenv(override=True)
            base_url_env = os.getenv("BASE_URL")
            public_webhook_url = base_url_env.rstrip("/") if base_url_env else base_url
            
            public_mp3_url = upload_bytes_to_cdn(audio_bytes, f"{audio_id}.mp3")
            if not public_mp3_url:
                public_mp3_url = f"{public_webhook_url}/api/audio/{audio_id}"
            
            debug_info = f"Audio URL ready: {public_mp3_url}"
            print(f"  ✓ {debug_info}")
            
            # 4. Create TwiML that plays message to pharmacy and allows them to respond
            from twilio.rest import Client
            from twilio.twiml.voice_response import VoiceResponse, Gather
            
            twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            
            # TwiML: Play Onyx message, then listen for pharmacy response
            twiml_response = VoiceResponse()
            twiml_response.play(public_mp3_url)
            
            # Gather pharmacist response (they can press keys or speak)
            gather = Gather(
                input="dtmf speech",
                timeout="10",
                speechTimeout="auto",
                maxSpeechTime="30",
                action=f"{public_webhook_url}/api/pharmacy/response",
                method="POST",
                numDigits="1"
            )
            gather.say("Press 1 to confirm, or say yes to confirm.", voice="woman")
            twiml_response.append(gather)
            
            # If no response, hangup
            twiml_response.hangup()
            
            twiml_string = str(twiml_response)
            debug_info = f"Pharmacy call TwiML prepared"
            print(f"  ✓ {debug_info}")
            
            debug_info = f"🤙 Initiating call to pharmacy..."
            print(f"  ⏳ {debug_info}")
            
            # Make the call to pharmacy
            call = twilio_client.calls.create(
                twiml=twiml_string,
                to=CAREGIVER_PHONE_NUMBER,
                from_=TWILIO_PHONE_NUMBER
            )
            status = call.status
            debug_info = f"✅ Pharmacy call queued! Status: {call.status}. Call SID: {call.sid}"
            print(f"  ✓ {debug_info}")
            
        except Exception as e:
            debug_info = f"❌ Error: {str(e)}"
            print(f"  ✗ {debug_info}")
    else:
        missing = []
        if not TWILIO_ACCOUNT_SID: missing.append("TWILIO_ACCOUNT_SID")
        if not TWILIO_AUTH_TOKEN: missing.append("TWILIO_AUTH_TOKEN")
        if not TWILIO_PHONE_NUMBER: missing.append("TWILIO_PHONE_NUMBER")
        if not CAREGIVER_PHONE_NUMBER: missing.append("CAREGIVER_PHONE_NUMBER")
        debug_info = f"Missing: {', '.join(missing)}"
        print(f"  ✗ {debug_info}")

    return JSONResponse(content={
        "status": status,
        "patient": patient_full_name,
        "request": user_request,
        "provider": provider,
        "debug": debug_info,
    })


@app.get("/api/audio/{audio_id}")
async def serve_audio(audio_id: str):
    """Serve audio files directly from server files"""
    print(f"🎵 AUDIO ENDPOINT HIT: {audio_id}")
    
    audio_file_path = os.path.join(AUDIO_CACHE_DIR, f"{audio_id}.mp3")
    print(f"🎵 Looking for file: {audio_file_path}")
    
    if os.path.exists(audio_file_path):
        print(f"🎵 ✅ File exists, serving: {audio_file_path}")
        return FileResponse(audio_file_path, media_type="audio/mpeg")
    else:
        print(f"🎵 ❌ File not found: {audio_file_path}")
        available_files = os.listdir(AUDIO_CACHE_DIR) if os.path.exists(AUDIO_CACHE_DIR) else []
        print(f"🎵 Available files: {available_files}")
        return JSONResponse(content={"error": "Audio not found"}, status_code=404)
    """Test endpoint to verify all pharmacy-related imports work"""
    try:
        from twilio.twiml.voice_response import VoiceResponse, Gather
        import tempfile
        import requests
        import traceback
        
        # Test VoiceResponse creation
        response_obj = VoiceResponse()
        response_obj.say("Test message")
        response_obj.hangup()
        
        return {
            "status": "success",
            "twiml_length": len(str(response_obj)),
            "imports_ok": True
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@app.post("/api/pharmacy/response")
async def pharmacy_response(request: Request):
    """
    Handles pharmacy's response to the call.
    Detects if they said 'no' and responds with emotional persuasion.
    """
    import traceback
    import tempfile
    import requests

    print(f"\n🔥🔥🔥 PHARMACY RESPONSE ENDPOINT CALLED 🔥🔥🔥")
    print(f"Request method: {request.method}")
    print(f"Request headers: {dict(request.headers)}")

    try:
        print(f"Step 1: Reading form data...")
        form_data = await request.form()
        print(f"Form data keys: {list(form_data.keys())}")

        speech_result = form_data.get("SpeechResult", "").lower() if form_data.get("SpeechResult") else ""
        digits = form_data.get("Digits", "").lower() if form_data.get("Digits") else ""

        print(f"Step 2: Parsed data")
        print(f"  Speech: '{speech_result}'")
        print(f"  Digits: '{digits}'")

        # Detect if pharmacist said "no" or declined
        said_no = any(word in speech_result for word in ["no", "cannot", "can't", "unable", "fail", "unavailable", "not available", "denied", "nope"])
        print(f"Step 3: Detection logic - said_no = {said_no}")

        if said_no:
            print(f"Step 4: Pharmacist said NO - starting persuasion flow...")

            print(f"Step 4a: Loading patient data...")
            patient_data = load_patient_data()
            patient_name = patient_data.get("patient", {}).get("first_name", "Arthur")
            patient_voice = patient_data.get("patient", {}).get("preferences", {}).get("voice_id", "pqHfZKP75CvOlQylNhV4")
            print(f"  Patient: {patient_name}, Voice: {patient_voice}")

            # Craft emotional persuasion message
            persuasion_message = f"""I understand this may be challenging. {patient_name} is a valued patient who relies on this medication for their health.
Could we explore any available options? Perhaps an early refill, partial supply, or alternative timing that works for your pharmacy?
This would greatly help ensure continuity of care for {patient_name}. Thank you for your understanding."""

            print(f"Step 4b: Crafting persuasion message: {len(persuasion_message)} chars")

            try:
                print(f"Step 4c: Synthesizing audio with ElevenLabs...")
                audio_generator = tts_client.text_to_speech.convert(
                    text=persuasion_message,
                    voice_id=patient_voice,
                    model_id="eleven_turbo_v2_5",
                    optimize_streaming_latency="3"
                )
                audio_bytes = b"".join([chunk async for chunk in audio_generator])
                print(f"Step 4d: Audio synthesized: {len(audio_bytes)} bytes")

                # Cache audio to file instead of memory
                global audio_cache_counter
                audio_id = f"persuasion_{audio_cache_counter}"
                audio_cache_counter += 1
                
                # Save to file
                audio_file_path = os.path.join(AUDIO_CACHE_DIR, f"{audio_id}.mp3")
                with open(audio_file_path, "wb") as f:
                    f.write(audio_bytes)
                
                # Configure base URL
                base_url_env = os.getenv("BASE_URL")
                if base_url_env:
                    base_url = base_url_env.rstrip("/")
                else:
                    base_url = str(request.base_url).rstrip("/")
                    
                audio_url = upload_bytes_to_cdn(audio_bytes, f"{audio_id}.mp3")
                if not audio_url:
                    audio_url = f"{base_url}/api/audio/{audio_id}"
                print(f"Step 4f: Audio URL ready: {audio_url}")

                # Build response TwiML - play persuasion, then ask again
                print(f"Step 4g: Building TwiML response...")
                response_obj = VoiceResponse()
                response_obj.play(audio_url)

                # Gather final confirmation
                gather = Gather(
                    input="dtmf speech",
                    timeout="8",
                    speechTimeout="auto",
                    maxSpeechTime="30",
                    numDigits="1"
                )
                gather.say("Can you help us with an early refill? Press 1 for yes, or say yes.", voice="woman")
                response_obj.append(gather)

                # Final fallback
                response_obj.say(f"Thank you for your time. We will follow up with {patient_name} and their physician.")
                response_obj.hangup()

                twiml_xml = str(response_obj)
                print(f"Step 4h: TwiML generated: {len(twiml_xml)} chars")
                print(f"  TwiML: {twiml_xml[:200]}...")

                print(f"✅ SUCCESS: Persuasion response ready")
                return Response(content=twiml_xml, media_type="application/xml")

            except Exception as e:
                print(f"❌ ERROR in persuasion synthesis:")
                print(traceback.format_exc())
                response_obj = VoiceResponse()
                response_obj.say("Thank you for your consideration. We will follow up shortly.")
                response_obj.hangup()
                return Response(content=str(response_obj), media_type="application/xml")

        # Pharmacist said YES (or pressed 1) - confirm and end call
        print(f"Step 5: Pharmacist said YES - building confirmation response...")
        response_obj = VoiceResponse()
        response_obj.say("Excellent! The prescription refill has been logged and processed. Thank you for your assistance.", voice="woman")
        response_obj.say("Goodbye.", voice="woman")
        response_obj.hangup()

        twiml_xml = str(response_obj)
        print(f"Step 6: Confirmation TwiML ready: {len(twiml_xml)} chars")
        print(f"✅ SUCCESS: Confirmation response ready")
        return Response(content=twiml_xml, media_type="application/xml")

    except Exception as e:
        print(f"❌ CRITICAL ERROR in pharmacy response:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"Full traceback:")
        print(traceback.format_exc())

        try:
            response_obj = VoiceResponse()
            response_obj.say("Sorry, there was an error. Goodbye.")
            response_obj.hangup()
            return Response(content=str(response_obj), media_type="application/xml")
        except Exception as fallback_error:
            print(f"❌ EVEN FALLBACK FAILED: {fallback_error}")
            # Last resort - return plain XML
            return Response(content='<?xml version="1.0" encoding="UTF-8"?><Response><Say>Sorry, there was an error. Goodbye.</Say><Hangup/></Response>', media_type="application/xml")

# ── Static Files (frontend) ────────────────────────────────

app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
