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
JSON_PATH = os.path.join(BASE_DIR, "OCR", "output", "report.json")

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
    valid_exts = (".pdf", ".png", ".jpg", ".jpeg")
    if not any(file.filename.lower().endswith(ext) for ext in valid_exts):
        raise HTTPException(status_code=400, detail="Only PDF and image lab reports (.pdf, .png, .jpg, .jpeg) are supported.")

    try:
        # Purge stale JSON output before processing new file
        if os.path.exists(JSON_PATH):
            os.remove(JSON_PATH)

        # Clear old files safely in input folder
        for old_f in os.listdir(INPUT_DIR):
            old_path = os.path.join(INPUT_DIR, old_f)
            if os.path.isfile(old_path):
                try:
                    os.remove(old_path)
                except OSError:
                    pass

        # Save uploaded PDF with sanitized filename
        safe_filename = os.path.basename(file.filename)
        saved_pdf_path = os.path.join(INPUT_DIR, safe_filename)
        with open(saved_pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Execute model pipeline subprocess
        proc = subprocess.run(
            [sys.executable, os.path.join(BASE_DIR, "load_models.py"), safe_filename],
            capture_output=True,
            text=True
        )

        # Validate that fresh JSON was generated
        if not os.path.exists(JSON_PATH):
            raise HTTPException(
                status_code=500, 
                detail=f"OCR parsing failed. Subprocess output: {proc.stderr or proc.stdout}"
            )

        with open(JSON_PATH, "r", encoding="utf-8") as f:
            raw_ocr_output = json.load(f)

        return JSONResponse(content={
            "success": True,
            "filename": safe_filename,
            "raw_summary_output": proc.stdout,
            "parsed_data": raw_ocr_output
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnostic Pipeline Error: {str(e)}")