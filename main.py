import os
import tempfile
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import pytesseract
from ultralytics import YOLO
import requests
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model = YOLO("yolov8s.pt")


GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Please set GEMINI_API_KEY environment variable")


def detect_and_ocr(image_path: str, lang: str = "eng") -> Dict[str, Any]:
    """
    Runs YOLOv8 object detection and Tesseract OCR on the provided image.
    Returns a dict with:
      - text: OCR-extracted text
      - detections: list of { class, confidence }
    """
    # 1) YOLOv8 inference
    results = model(image_path, show=False)
    det = results[0]

    detections = []
    if hasattr(det, "boxes") and det.boxes is not None:
        for box in det.boxes:
            cls_id = int(box.cls.cpu().numpy()) if hasattr(box, "cls") else None
            conf = float(box.conf.cpu().numpy()) if hasattr(box, "conf") else None
            cls_name = model.names[cls_id] if cls_id in model.names else str(cls_id)
            detections.append({"class": cls_name, "confidence": round(conf, 2)})

    # 2) Tesseract OCR
    img = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(img, lang=lang).strip()

    return {
        "text": extracted_text,
        "detections": detections,
    }


def call_gemini(text: str, detections: Any) -> str:
    """
    Builds a prompt containing the raw OCR text and YOLO detections,
    sends it to the Gemini API, and returns the summary text.
    """
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    )
    headers = {"Content-Type": "application/json"}

  
    if detections:
        items = [f"{d['class']} ({d['confidence']})" for d in detections]
        detection_summary = "Detected objects: " + ", ".join(items) + "."
    else:
        detection_summary = "No objects detected."

    
    prompt_text = (
        "You are an image-analysis assistant. "
        "Below is the OCR-extracted text from an uploaded image, "
        "and a summary of all detected objects. "
        "Provide a long extended detailed with all details, human-readable summary of what’s in this image.\n\n"
        f"OCR text:\n\"\"\"\n{text}\n\"\"\"\n\n"
        f"{detection_summary}\n\n"
        "Respond with  paragraph describing the content of the image."
    )

    payload = {
        "contents": [
            {"parts": [{"text": prompt_text}]}
        ]
    }

    resp = requests.post(url, headers=headers, json=payload)
    if resp.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=f"Gemini API error {resp.status_code}: {resp.text}",
        )

    response_json = resp.json()
    try:
       
        generated = response_json["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected Gemini response format: {response_json}"
        )

    return generated.strip()


@app.post("/Image-analysis")
async def upload_and_summarize(

    file: UploadFile = File(...),
    lang: str = "eng",
):
   
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

  
    try:
        result = detect_and_ocr(tmp_path, lang)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection/OCR error: {e}")

   
    try:
        summary_text = call_gemini(result["text"], result["detections"])
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini call error: {e}")

   
    return JSONResponse(content={"summary": summary_text})

