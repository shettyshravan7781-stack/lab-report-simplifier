import os
import shutil
import json
import sys
import subprocess
import glob
import re
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(
    title="AI Medical Lab Report Analyzer API",
    description="Vision-LLM Engine with Gemini Automatic Rate-Limit Retry",
    version="1.0.0"
)

BASE_DIR = r"C:\AI_Lab_Report"
INPUT_DIR = os.path.join(BASE_DIR, "input")
JSON_OUTPUT_PATH = os.path.join(BASE_DIR, "OCR", "output", "report_output.json")
LOAD_MODELS_SCRIPT = os.path.join(BASE_DIR, "load_models.py")

SKIP_METADATA_KEYWORDS = [
    "REGISTERED ON", "COLLECTED ON", "REPORTED ON", "PATIENT NAME",
    "REF BY", "SAMPLE TYPE", "AGE / GENDER", "AGE", "GENDER", "DATE"
]

def auto_patch_deprecated_models():
    """Ensures deprecated or decommissioned Gemini model strings are updated to active models."""
    try:
        py_files = glob.glob(os.path.join(BASE_DIR, "**", "*.py"), recursive=True)
        for filepath in py_files:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            
            # Patch decommissioned gemini-3.6-flash to current gemini-3.6-flash
            updated_content = content.replace("gemini-3.6-flash", "gemini-3.6-flash")
            updated_content = updated_content.replace("gemini-3.6-flash", "gemini-3.6-flash")
            
            if updated_content != content:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(updated_content)
    except Exception as e:
        print(f"[WARN] Model patching warning: {e}")

def clean_json_payload(data):
    """Filter out non-biomarker header rows before returning response."""
    if isinstance(data, dict) and "tests" in data and isinstance(data["tests"], list):
        cleaned_tests = []
        for t in data["tests"]:
            test_name = str(t.get("test", "")).upper().strip()
            if not any(kw in test_name for kw in SKIP_METADATA_KEYWORDS):
                cleaned_tests.append(t)
        data["tests"] = cleaned_tests
    return data

def parse_retry_delay(logs: str) -> float:
    """Extracts required retry delay in seconds from Gemini 429 error messages."""
    match = re.search(r"Please retry in (\d+(?:\.\d+)?)s", logs)
    if match:
        return float(match.group(1)) + 1.0  # Add 1s buffer
    return 15.0  # Default pause if regex match is not found

@app.get("/", tags=["Health Check"])
def root():
    return {"status": "Active", "message": "Clinical Diagnostic API is running."}

@app.post("/analyze/", tags=["Report Analysis"])
async def analyze_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    auto_patch_deprecated_models()

    os.makedirs(INPUT_DIR, exist_ok=True)
    file_path = os.path.join(INPUT_DIR, file.filename)
    
    try:
        # Save incoming PDF file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        max_attempts = 3
        process = None

        for attempt in range(max_attempts):
            process = subprocess.run(
                [sys.executable, LOAD_MODELS_SCRIPT],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env
            )

            logs = process.stdout or ""
            
            # Check if Gemini hit a 429 rate limit
            if "429 RESOURCE_EXHAUSTED" in logs and attempt < max_attempts - 1:
                wait_time = parse_retry_delay(logs)
                print(f"[INFO] Gemini 429 Rate Limit hit. Waiting {wait_time:.1f}s for quota reset (Attempt {attempt + 1}/{max_attempts})...")
                time.sleep(wait_time)
                continue
            
            break

        if os.path.exists(JSON_OUTPUT_PATH):
            with open(JSON_OUTPUT_PATH, "r", encoding="utf-8") as f:
                output_data = json.load(f)
            
            cleaned_data = clean_json_payload(output_data)

            return JSONResponse(content={
                "status": "success",
                "filename": file.filename,
                "data": cleaned_data,
                "terminal_logs": process.stdout
            })
        else:
            raise HTTPException(status_code=500, detail="Pipeline executed, but report output JSON was not found.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)