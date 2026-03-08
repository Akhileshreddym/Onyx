
Introducing Onyx,
An autonomous, medical proxy that cross references patient records to prevent fatal drug interactions via real time Phone Calls.

 ![alt text](image.png)

Core Features:
-> Instant Dossier: Uses Gemini 2.0 to parse unstructured Medical PDF’s into JSON profiles.
-> Real-Time Telephony: Full-duplex Twilio WebSockets bridging web clients to live phone lines
-> Native Voice Synthesis: Leverages ElevenLabs Multilingual v3 and custom acoustic tuning for sub-second, hyper-realistic conversational AI
-> Autonomous Safety Net: Dynamically halts prescriptions that conflict with known patient allergies mid-conversation
-> Visual Pill Scanner: Patients can hold any medication bottle up to their webcam. Our vision model identifies the exact drug, cross-references it with their medical profile, and immediately sounds an alarm if a dangerous **allergy conflict** is detected.

FlowChart:
Frontend -> Backend -> Brain(Gemini) -> Voice(Elevenlabs) -> Telephony(Twilio)

Tech Stack:
Frontend: HTML, Tailwind CSS, GSAP
Backend: Python, FastAPI, WebSockets
AI & Data: Gemini 1.5 Flash, OpenRouter
Voice & Comm: ElevenLabs, Twilio



## ⚙️ Running Onyx Locally

### Prerequisites
*   Python 3.9+
*   API Keys for **Twilio**, **ElevenLabs**, **OpenRouter**, and **Gemini**.

### Installation
1.  **Clone the repo**
    ```bash
    git clone https://github.com/your-org/onyx.git
    cd onyx/backend
    ```
2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Environment Variables**
    Rename `.env.example` to `.env` and fill in your keys:
    ```env
    TWILIO_ACCOUNT_SID=...
    TWILIO_AUTH_TOKEN=...
    TWILIO_PHONE_NUMBER=...
    CAREGIVER_PHONE_NUMBER=...
    ELEVENLABS_API_KEY=...
    OPENROUTER_API_KEY=...
    OPENAI_API_KEY=...
    ```
4.  **Start the Server & Tunnel**
    Run the setup script which boots FastAPI and the SSH tunnel for Twilio:
    ```bash
    python3 start_tunnel.py
    ```
5.  **Access the Dashboard**
    Open `http://localhost:8000/onboard` in your browser.
