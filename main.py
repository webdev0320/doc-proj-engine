import os
# STABILITY FLAGS
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['PADDLE_ONEDNN'] = 'OFF'
os.environ['FLAGS_enable_new_executor'] = '0'
os.environ['FLAGS_enable_new_ir'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'

import shutil
import requests
import cv2
import numpy as np
import json
import fitz
import paramiko
import boto3
from io import BytesIO
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import easyocr
from paddleOCR import classify_page
from crypto_utils import decrypt_file
from dotenv import load_dotenv

load_dotenv()

# Initialize EasyOCR once at startup
print("Loading EasyOCR model...")
reader = easyocr.Reader(['en'], gpu=False)
print("EasyOCR ready.")

app = FastAPI(title="IDP Engine - PaddleOCR")

@app.middleware("http")
async def log_requests(request, call_next):
    import time
    start_time = time.time()
    path = request.url.path
    method = request.method
    print(f">>> [DEBUG] Incoming: {method} {path}")
    response = await call_next(request)
    process_time = time.time() - start_time
    print(f"<<< [DEBUG] Outgoing: {method} {path} - Status: {response.status_code} ({process_time:.4f}s)")
    return response

@app.get("/")
def root():
    return {"status": "running", "endpoints": ["/health", "/process", "/export"]}


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:3001")
STORAGE_DIR = os.path.join(os.path.dirname(__file__), "../storage")
BLOBS_DIR = os.path.join(STORAGE_DIR, "blobs")
PAGES_DIR = os.path.join(STORAGE_DIR, "pages")

# Ensure directories exist
os.makedirs(BLOBS_DIR, exist_ok=True)
os.makedirs(PAGES_DIR, exist_ok=True)

class ProcessRequest(BaseModel):
    blob_id: str
    storage_path: str
    storage_settings: Optional[Dict[str, Any]] = None

class ExportRequest(BaseModel):
    blob_id: str
    filename: str
    manifest: list # list of {documentId, documentName, pages: [s3Path]}
    storage_settings: Optional[Dict[str, Any]] = None

@app.get("/health")
def health():
    return {"status": "ok", "service": "idp-engine"}

def download_from_remote(filename, local_path, settings):
    """Downloads a file from S3 or SFTP based on provided settings."""
    provider = settings.get('provider', 'SFTP')
    
    if provider == 'S3':
        bucket = settings.get('s3Bucket')
        access_key = settings.get('s3AccessKey')
        secret_key = settings.get('s3SecretKey')
        region = settings.get('s3Region', 'us-east-1')
        
        if not all([bucket, access_key, secret_key]):
            raise ValueError("Missing S3 credentials in settings")
            
        s3 = boto3.client('s3', region_name=region, aws_access_key_id=access_key, aws_secret_access_key=secret_key)
        try:
            s3.download_file(bucket, f"Inbound/{filename}", local_path)
        except:
            print(f"File not in Inbound. Checking Archive for {filename}...")
            s3.download_file(bucket, f"Archive/{filename}", local_path)
    else:
        # SFTP
        host = settings.get('sftpHost')
        port = int(settings.get('sftpPort', 22))
        user = settings.get('sftpUser')
        password = settings.get('sftpPass')
        
        if not all([host, user, password]):
            raise ValueError("Missing SFTP credentials in settings")
            
        transport = paramiko.Transport((host, port))
        transport.connect(username=user, password=password)
        sftp = paramiko.SFTPClient.from_transport(transport)
        try:
            sftp.get(f"/Inbound/{filename}", local_path)
        except:
            print(f"File not in Inbound. Checking Archive for {filename}...")
            sftp.get(f"/Archive/{filename}", local_path)
        sftp.close()
        transport.close()

def upload_to_remote(local_path, remote_path, settings):
    """Uploads a file to S3 or SFTP based on provided settings."""
    provider = settings.get('provider', 'SFTP')
    if provider == 'S3':
        bucket = settings.get('s3Bucket')
        access_key = settings.get('s3AccessKey')
        secret_key = settings.get('s3SecretKey')
        region = settings.get('s3Region', 'us-east-1')
        if all([bucket, access_key, secret_key]):
            s3 = boto3.client('s3', region_name=region, aws_access_key_id=access_key, aws_secret_access_key=secret_key)
            s3.upload_file(local_path, bucket, remote_path)
    else:
        host, user, p = settings.get('sftpHost'), settings.get('sftpUser'), settings.get('sftpPass')
        port = int(settings.get('sftpPort', 22))
        if all([host, user, p]):
            transport = paramiko.Transport((host, port))
            transport.connect(username=user, password=p)
            sftp = paramiko.SFTPClient.from_transport(transport)
            # Ensure directory exists (e.g., /pages)
            remote_dir = os.path.dirname(remote_path)
            try: sftp.mkdir(remote_dir)
            except: pass
            sftp.put(local_path, remote_path)
            sftp.close()
            transport.close()

@app.post("/process")
async def process_document(req: ProcessRequest, background_tasks: BackgroundTasks):
    pdf_path = os.path.join(BLOBS_DIR, req.storage_path)
    
    # If file doesn't exist locally or is empty (failed previous download), try to download it
    if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) == 0:
        if os.path.exists(pdf_path):
            print(f"Empty file found at {pdf_path}. Deleting...")
            os.remove(pdf_path)
            
        if req.storage_settings:
            try:
                # UUIDs are 36 chars. Backend uses f"{uuid}-{originalName}"
                # So we skip 36 chars + 1 hyphen = 37 chars
                remote_filename = req.storage_path[37:] if (len(req.storage_path) > 37 and req.storage_path[36] == '-') else req.storage_path
                print(f"File missing locally. Downloading {remote_filename} from remote...")
                download_from_remote(remote_filename, pdf_path, req.storage_settings)
            except Exception as e:
                print(f"[ERROR] Remote download failed: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to download file from remote storage: {e}")
        else:
            raise HTTPException(status_code=404, detail=f"PDF not found locally and no storage settings provided.")

    # Run OCR in background
    background_tasks.add_task(run_pipeline, req.blob_id, pdf_path, req.storage_settings)
    
    return {"success": True, "message": "OCR processing started in background"}

def scan_health_check(image_path):
    """Detects skew, brightness, and blur using OpenCV."""
    img = cv2.imread(image_path)
    if img is None: return [], 0, 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_blurry = fm < 100
    avg_brightness = np.mean(gray)
    is_dark = avg_brightness < 40
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    skew_angle = 0
    if lines is not None:
        angles = [(line[0][1] * 180 / np.pi) - 90 for line in lines[:10]]
        skew_angle = np.median(angles)
    issues = []
    if is_blurry: issues.append("BLURRY")
    if is_dark: issues.append("TOO_DARK")
    if abs(skew_angle) > 5: issues.append("SKEWED")
    return issues, fm, skew_angle

def extract_mortgage_entities(ocr_text, doc_type):
    data = {}
    full_text = ocr_text.upper() if isinstance(ocr_text, str) else ""
    if "W-2" in doc_type or "W2" in doc_type:
        data = {"Employer_EIN": "12-3456789", "Box1_Wages": "85,400.00"}
    elif "PAYSTUB" in doc_type:
        data = {"Period_End": "2026-03-31", "YTD_Gross": "21,350.50"}
    elif "1040" in doc_type:
        data = {"AGI": "142,000.00", "Tax_Year": "2025"}
    return data

@app.post("/export")
async def export_documents(req: ExportRequest):
    try:
        exported_files = []
        export_folder = os.path.join(STORAGE_DIR, "final", req.blob_id)
        os.makedirs(export_folder, exist_ok=True)
        
        for item in req.manifest:
            doc_name_clean = "".join([c for c in item['documentName'] if c.isalnum() or c in (' ', '-', '_')]).strip()
            output_filename = f"{doc_name_clean}.pdf"
            output_path = os.path.join(export_folder, output_filename)
            
            new_pdf = fitz.open() 
            for page_s3_path in item['pages']:
                img_path = os.path.join(PAGES_DIR, page_s3_path)
                if os.path.exists(img_path):
                    img_doc = fitz.open(img_path)
                    pdf_bytes = img_doc.convert_to_pdf()
                    img_doc.close()
                    temp_doc = fitz.open("pdf", pdf_bytes)
                    new_pdf.insert_pdf(temp_doc)
                    temp_doc.close()
            
            new_pdf.save(output_path)
            new_pdf.close()
            exported_files.append(output_filename)
        
        print(f"[SUCCESS] Exported {len(exported_files)} PDFs for Blob {req.blob_id}")

        # --- DYNAMIC STORAGE UPLOAD LOGIC ---
        settings = req.storage_settings or {}
        provider = settings.get('provider', 'SFTP')
        
        log_path = os.path.join(STORAGE_DIR, "engine_sftp.log")
        with open(log_path, "a") as logf:
            logf.write(f"\n[{req.blob_id}] Starting {provider} Upload Logic.\n")
            
            if provider == 'S3':
                bucket, ak, sk = settings.get('s3Bucket'), settings.get('s3AccessKey'), settings.get('s3SecretKey')
                region = settings.get('s3Region', 'us-east-1')
                if all([bucket, ak, sk]):
                    try:
                        s3 = boto3.client('s3', region_name=region, aws_access_key_id=ak, aws_secret_access_key=sk)
                        for f in exported_files:
                            s3.upload_file(os.path.join(export_folder, f), bucket, f"Outbound/{f}")
                        logf.write(f"[{req.blob_id}] S3 Upload completed.\n")
                    except Exception as e: logf.write(f"[{req.blob_id}] S3 Error: {e}\n")
            else: # SFTP
                host, user, p = settings.get('sftpHost'), settings.get('sftpUser'), settings.get('sftpPass')
                port = int(settings.get('sftpPort', 22))
                if all([host, user, p]):
                    try:
                        transport = paramiko.Transport((host, port))
                        transport.connect(username=user, password=p)
                        sftp = paramiko.SFTPClient.from_transport(transport)
                        try: sftp.mkdir("/Outbound")
                        except: pass
                        for f in exported_files:
                            sftp.put(os.path.join(export_folder, f), f"/Outbound/{f}")
                        sftp.close()
                        transport.close()
                        logf.write(f"[{req.blob_id}] SFTP Upload completed.\n")
                    except Exception as e: logf.write(f"[{req.blob_id}] SFTP Error: {e}\n")

        return {"success": True, "files": exported_files}
    except Exception as e:
        print(f"[ERROR] Export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def run_pipeline(blob_id: str, pdf_path: str, storage_settings: Optional[Dict[str, Any]] = None):
    try:
        print(f">>> [PIPELINE] Starting for Blob: {blob_id}")
        print(f">>> [PIPELINE] Input path: {pdf_path}")
        
        if not os.path.exists(pdf_path):
             print(f"!!! [PIPELINE ERROR] File does not exist: {pdf_path}")
             return

        # Skipping decryption as files are currently unencrypted
        print(f">>> [PIPELINE] Opening PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        print(f">>> [PIPELINE] PDF opened. Pages: {len(doc)}")
        page_records = []
        for i in range(len(doc)):
            page_num = i + 1
            page = doc[i]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            image_filename = f"{blob_id}_{page_num}.png"
            image_path = os.path.join(PAGES_DIR, image_filename)
            pix.save(image_path)
            
            # Proactively upload to remote storage so Vercel can see it
            if storage_settings:
                try:
                    remote_image_path = f"pages/{image_filename}"
                    print(f">>> [PIPELINE] Uploading page {page_num} to remote...")
                    upload_to_remote(image_path, remote_image_path, storage_settings)
                except Exception as ue:
                    print(f"!!! [PIPELINE ERROR] Page upload failed: {ue}")
            
            anomalies, blur_score, skew = scan_health_check(image_path)
            pdf_text = page.get_text("text").strip()
            
            try:
                if len(pdf_text) > 20:
                    ocr_text = pdf_text
                    res = "NativeTextExtractor"
                else:
                    res = reader.readtext(image_path)
                    ocr_text = " ".join([item[1] for item in res]) if res else ""
            except:
                ocr_text, res = "", None

            text_blocks = [{'text': item[1], 'confidence': item[2]} for item in res] if (res and res != "NativeTextExtractor") else [{'text': ocr_text, 'confidence': 1.0}]
            ai_label, fuzzy_confidence = classify_page(text_blocks)
            confidence = fuzzy_confidence if res else max(fuzzy_confidence - 0.1, 0.0)
            should_flag = confidence < 0.85 or len(anomalies) > 0
            extracted = extract_mortgage_entities(ocr_text, ai_label)
            
            page_records.append({
                "page_index": i, "s3_path": image_filename, "ai_label": ai_label,
                "confidence_score": confidence, "is_flagged": should_flag,
                "anomaly_flags": json.dumps(anomalies), "extracted_data": json.dumps(extracted)
            })
            print(f"Page {page_num}: {ai_label} (Conf: {confidence})")

        requests.post(f"{BACKEND_URL}/api/blobs/{blob_id}/pages", json={"pages": page_records})
        requests.patch(f"{BACKEND_URL}/api/blobs/{blob_id}", json={"status": "COMPLETED"})
        doc.close()
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        requests.patch(f"{BACKEND_URL}/api/blobs/{blob_id}", json={"status": "FAILED"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
