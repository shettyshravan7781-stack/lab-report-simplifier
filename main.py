import os
import shutil
import subprocess
import sys
import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# 1. Initialize FastAPI App
app = FastAPI(
    title="AI Clinical Diagnostic Pipeline API",
    description="Backend API for Lab Report OCR, XGBoost Organ Risk Analysis, and Pathology Narratives",
    version="1.0.0"
)

# 2. Enable CORS (Crucial so frontend on localhost:3000 / localhost:5173 can call this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production to frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input")
JSON_PATH = os.path.join(BASE_DIR, "OCR", "output", "report_output.json")

os.makedirs(INPUT_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# ENDPOINT 1: Health Check (To verify backend is running)
# -------------------------------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "online", "message": "AI Clinical Diagnostic Pipeline API is running."}

# -------------------------------------------------------------------------
# ENDPOINT 2: Upload PDF & Process Full Clinical Diagnostic Audit
# -------------------------------------------------------------------------
@app.post("/api/analyze-report")
async def analyze_report(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF lab reports are supported.")

    try:
        # Clear old PDFs in input folder
        for old_f in os.listdir(INPUT_DIR):
            old_path = os.path.join(INPUT_DIR, old_f)
            if os.path.isfile(old_path):
                os.remove(old_path)

        # Save uploaded PDF file to input/ directory
        saved_pdf_path = os.path.join(INPUT_DIR, file.filename)
        with open(saved_pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Import load_models execution functions dynamically or trigger CLI execution
        # Running load_models as subprocess ensures clean execution and cache flush
        proc = subprocess.run(
            [sys.executable, os.path.join(BASE_DIR, "load_models.py")],
            capture_output=True,
            text=True
        )

        # Read generated OCR output JSON
        if not os.path.exists(JSON_PATH):
            raise HTTPException(status_code=500, detail="OCR parsing failed to produce diagnostic JSON output.")

        with open(JSON_PATH, "r", encoding="utf-8") as f:
            raw_ocr_output = json.load(f)

        # Return structured response payload for Frontend Rendering
        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "raw_summary_output": proc.stdout,
            "parsed_data": raw_ocr_output
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnostic Pipeline Error: {str(e)}")