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


# ── API Routes ──────────────────────────────────────────────

@app.get("/api/patient")
async def get_patient():
    """Return full patient dossier."""
    return JSONResponse(content=load_patient_data())


@app.post("/api/scan")
async def scan_medication():
    """
    Stub: returns a pill-scan result with an allergy conflict alert.
    In production this would accept an image and run Gemini Vision.
    """
    patient = load_patient_data()
    allergies = [a["substance"] for a in patient.get("allergies", [])]

    scanned_drug = "Amoxicillin 500mg"
    drug_group = "Penicillin Group"
    conflict = "Penicillin" in allergies

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
    """
    Stub: confirms an emergency SMS dispatch via Twilio.
    In production this would call the Twilio API.
    """
    patient = load_patient_data()
    caregiver = patient.get("caregiver", {})

    return JSONResponse(content={
        "status": "DELIVERED",
        "recipient": caregiver.get("name", "Unknown"),
        "relationship": caregiver.get("relationship", "Unknown"),
        "message": "Emergency SMS sent to Daughter: Pharmacy out of stock / Allergy detected",
        "provider": "Twilio",
    })


# ── Static Files (frontend) ────────────────────────────────

app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
