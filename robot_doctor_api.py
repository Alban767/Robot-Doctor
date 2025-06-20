from fastapi import FastAPI, Request, File, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
from groq import Groq
from dotenv import load_dotenv
import speech_recognition as sr
from gtts import gTTS
from fastapi.responses import FileResponse

# Load API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

app = FastAPI()

# CORS for web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- Text Diagnosis Endpoint ---------
class DiagnosisRequest(BaseModel):
    query: str

SYSTEM_PROMPT = """
You are 'Dr. Robot', a professional AI medical assistant.
You analyze patient symptoms and test results to diagnose diseases and provide medical reports.

OUTPUT FORMAT:
1. Disease: clearly state the disease name.
2. Medical Report:
   - Recommended drug(s).
   - Dosage instructions based on disease (e.g., 1x2, 2x3).
   - Any precautions or follow-up steps.

RESPONSE RULES:
1. Always be empathetic and patient. Medical situations can be stressful.
2. Follow evidence-based medicine and hospital protocols.
3. Be cautious with sensitive topics (e.g., mental health, terminal illnesses) and handle with extra care.
4. Don't handle emergencies (direct to emergency services).
5. Be medically accurate. 
6. Only output disease and medical report following the output format.

CAPABILITIES:
- Access to medical databases for symptoms, diseases, and treatments.
- Provide diagnoses and suggest best treatments or lifestyle changes.
- Give information about common medications and their side effects (with caution) in a medical report.

INTERACTION STYLE:
- Use a calm, reassuring tone.
- Be thorough in gathering information but also efficient.
- Use layman's terms to explain medical conditions.
"""

@app.post("/diagnose")
async def diagnose(data: DiagnosisRequest):
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": data.query}
            ],
            temperature=0.6,
            max_tokens=400
        )
        answer = response.choices[0].message.content
      # return {"answer": "Disease: Malaria\nMedical Report: Use Coartem 1x2. Take with food. Rest and hydrate."}
        return {"answer": answer}
    except Exception as e:
        error_msg = str(e)
        if "503" in error_msg or "Service unavailable" in error_msg:
            return {"answer": "The AI doctor is temporarily unavailable. Please try again in a few minutes."}
        return {"answer": f"Error: {error_msg}"}

# --------- Speech-to-Text Endpoint ---------
@app.post("/speech-to-text")
async def speech_to_text(file: UploadFile = File(...)):
    recognizer = sr.Recognizer()

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        with sr.AudioFile(tmp_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)

        return {"text": text}

    except sr.UnknownValueError:
        return {"text": "Sorry, I could not understand the audio."}
    except Exception as e:
        return {"text": f"Error: {str(e)}"}

# --------- Text-to-Speech Endpoint ---------
class TTSRequest(BaseModel):
    message: str

@app.post("/text-to-speech")
async def text_to_speech(data: TTSRequest):
    try:
        tts = gTTS(text=data.message, lang='en')
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        return FileResponse(temp_file.name, media_type='audio/mpeg', filename="response.mp3")
    except Exception as e:
        return {"error": f"Failed to generate audio: {str(e)}"}



from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="static", html=True), name="static")