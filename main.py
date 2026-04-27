import os
# STABILITY FLAGS
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['PADDLE_ONEDNN'] = 'OFF'
os.environ['FLAGS_enable_new_executor'] = '0'
os.environ['FLAGS_enable_new_ir'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'

import shutil
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import easyocr
from paddleOCR import classify_page

# Initialize EasyOCR once at startup (gpu=False for Windows CPU compatibility)
print("Loading EasyOCR model...")
reader = easyocr.Reader(['en'], gpu=False)
print("EasyOCR ready.")
import requests
import cv2
import numpy as np
import json
from crypto_utils import decrypt_file
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="IDP Engine - PaddleOCR")

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

class ExportRequest(BaseModel):
    blob_id: str
    filename: str
    manifest: list # list of {documentId, documentName, pages: [s3Path]}

@app.get("/health")
def health():
    return {"status": "ok", "service": "idp-engine"}

@app.post("/process")
async def process_document(req: ProcessRequest, background_tasks: BackgroundTasks):
    pdf_path = os.path.join(BLOBS_DIR, req.storage_path)
    
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail=f"PDF not found at {pdf_path}")

    # Run OCR in background to not block the request
    background_tasks.add_task(run_pipeline, req.blob_id, pdf_path)
    
    return {"success": True, "message": "OCR processing started in background"}

def scan_health_check(image_path):
    """Detects skew, brightness, and blur using OpenCV."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Blur Detection (Laplacian variance)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_blurry = fm < 100
    
    # 2. Brightness
    avg_brightness = np.mean(gray)
    is_dark = avg_brightness < 40
    
    # 3. Skew Detection (Simple version via HoughLines)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    skew_angle = 0
    if lines is not None:
        angles = []
        for line in lines[:10]:
            rho, theta = line[0]
            angle = (theta * 180 / np.pi) - 90
            angles.append(angle)
        skew_angle = np.median(angles)
    
    issues = []
    if is_blurry: issues.append("BLURRY")
    if is_dark: issues.append("TOO_DARK")
    if abs(skew_angle) > 5: issues.append("SKEWED")
    
    return issues, fm, skew_angle

def extract_mortgage_entities(ocr_text, doc_type):
    """Mock entity extraction based on document type."""
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
        import fitz
        exported_files = []
        export_folder = os.path.join(STORAGE_DIR, "final", req.blob_id)
        os.makedirs(export_folder, exist_ok=True)
        
        for item in req.manifest:
            # item: {documentId, documentName, pages: [s3Path]}
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
        import sqlite3
        import paramiko
        import boto3
        
        # Fetch settings directly from DB since engine is on same machine
        db_path = os.path.join(os.path.dirname(__file__), "../backend/prisma/dev.db")
        settings = {}
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM storage_settings WHERE id='default'")
            row = cursor.fetchone()
            if row:
                col_names = [description[0] for description in cursor.description]
                settings = dict(zip(col_names, row))
            conn.close()
        except Exception as e:
            print(f"[ERROR] Failed to fetch storage settings: {e}")
            
        provider = settings.get('provider', 'SFTP')
        
        with open(os.path.join(STORAGE_DIR, "engine_sftp.log"), "a") as logf:
            logf.write(f"\n[{req.blob_id}] Starting {provider} Upload Logic.\n")
            
            if provider == 'S3':
                bucket = settings.get('s3Bucket')
                access_key = settings.get('s3AccessKey')
                secret_key = settings.get('s3SecretKey')
                region = settings.get('s3Region', 'us-east-1')
                
                if bucket and access_key and secret_key:
                    try:
                        s3 = boto3.client(
                            's3',
                            region_name=region,
                            aws_access_key_id=access_key,
                            aws_secret_access_key=secret_key
                        )
                        for filename in exported_files:
                            local_path = os.path.join(export_folder, filename)
                            remote_path = f"Outbound/{filename}"
                            logf.write(f"[{req.blob_id}] Uploading {filename} to S3 {bucket}/{remote_path}...\n")
                            s3.upload_file(local_path, bucket, remote_path)
                        logf.write(f"[{req.blob_id}] S3 Upload completed successfully.\n")
                    except Exception as s3_e:
                        logf.write(f"[{req.blob_id}] [ERROR] S3 Upload failed: {s3_e}\n")
                else:
                    logf.write(f"[{req.blob_id}] [SKIPPED] Missing S3 credentials.\n")
                    
            else: # SFTP
                sftp_host = settings.get('sftpHost') or os.getenv("SFTP_HOST")
                sftp_port = settings.get('sftpPort') or int(os.getenv("SFTP_PORT", "22"))
                sftp_user = settings.get('sftpUser') or os.getenv("SFTP_USERNAME")
                sftp_pass = settings.get('sftpPass') or os.getenv("SFTP_PASSWORD")
                
                if sftp_host and sftp_user and sftp_pass:
                    try:
                        logf.write(f"[{req.blob_id}] Connecting to {sftp_host}:{sftp_port}...\n")
                        transport = paramiko.Transport((sftp_host, int(sftp_port)))
                        transport.connect(username=sftp_user, password=sftp_pass)
                        sftp = paramiko.SFTPClient.from_transport(transport)
                        
                        # Ensure /Outbound exists
                        try:
                            sftp.mkdir("/Outbound")
                        except IOError:
                            pass

                        # Upload directly to /Outbound (no subfolder)
                        remote_folder = "/Outbound"
                        
                        for filename in exported_files:
                            local_path = os.path.join(export_folder, filename)
                            remote_path = f"{remote_folder}/{filename}"
                            logf.write(f"[{req.blob_id}] Uploading {filename} to {remote_path}...\n")
                            sftp.put(local_path, remote_path)
                        
                        sftp.close()
                        transport.close()
                        logf.write(f"[{req.blob_id}] SFTP Upload completed successfully.\n")
                    except Exception as sftp_e:
                        logf.write(f"[{req.blob_id}] [ERROR] SFTP Upload failed: {sftp_e}\n")
                else:
                    logf.write(f"[{req.blob_id}] [SKIPPED] Missing SFTP credentials.\n")

        return {"success": True, "files": exported_files}
        
    except Exception as e:
        print(f"[ERROR] Export failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def run_pipeline(blob_id: str, pdf_path: str):
    try:
        print(f"Starting pipeline for Blob {blob_id}...")
        # 1. Decrypt raw PDF to temp file
        print(f"Decrypting {pdf_path}...")
        temp_pdf_path = f"{pdf_path}.decrypted"
        try:
            decrypt_file(pdf_path, temp_pdf_path)
            print(f"Decryption success. Temp file: {temp_pdf_path}")
        except Exception as e:
            print(f"[ERROR] Decryption failed: {str(e)}")
            requests.patch(f"{BACKEND_URL}/api/blobs/{blob_id}", json={"status": "FAILED"})
            return
            
        import fitz
        print(f"Opening PDF with fitz...")
        doc = fitz.open(temp_pdf_path)
        print(f"PDF opened. Page count: {len(doc)}")
        
        # 2. Iterate and save each page image
        page_records = []
        for i in range(len(doc)):
            page_num = i + 1
            page = doc[i]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) # 2x zoom for clarity
            
            image_filename = f"{blob_id}_{page_num}.png"
            image_path = os.path.join(PAGES_DIR, image_filename)
            
            pix.save(image_path)
            
            # --- PHASE 2: AI Enhancements ---
            anomalies, blur_score, skew = scan_health_check(image_path)
            
            # Fast-Path: Extract Native Embedded PDF Text (instantly handles forms like 1040s and W-2s)
            pdf_text = page.get_text("text").strip()
            
            try:
                if len(pdf_text) > 20:  # Native digital PDF — skip OCR
                    ocr_text = pdf_text
                    res = "NativeTextExtractor"
                    print(f"  [Page {page_num}] Native text ({len(pdf_text)} chars). Preview: {pdf_text[:120].replace(chr(10), ' ')!r}")
                else:
                    # Scanned page — run EasyOCR
                    print(f"  [Page {page_num}] No embedded text. Running EasyOCR...")
                    res = reader.readtext(image_path)  # returns [[bbox, text, confidence], ...]
                    ocr_text = " ".join([item[1] for item in res]) if res else ""
                    print(f"  [Page {page_num}] EasyOCR result ({len(ocr_text)} chars). Preview: {ocr_text[:120]!r}")
            except Exception as ocr_err:
                print(f"[WARNING] OCR failed on page {page_num}: {str(ocr_err)}")
                ocr_text = ""
                res = None

            # --- Rich fuzzy classification via paddleOCR.py ---
            if res and res != "NativeTextExtractor":
                # EasyOCR result: [[bbox, text, confidence], ...]
                text_blocks = [{'text': item[1], 'confidence': item[2]} for item in res]
            else:
                # Native PDF text or empty: wrap as a single block
                text_blocks = [{'text': ocr_text, 'confidence': 1.0}]

            ai_label, fuzzy_confidence = classify_page(text_blocks)
            print(f"  [Page {page_num}] classify_page -> label={ai_label}, fuzzy_score={fuzzy_confidence:.2f}")

            # Use real fuzzy score as confidence; penalise slightly if OCR itself failed
            confidence = fuzzy_confidence if res else max(fuzzy_confidence - 0.1, 0.0)
            should_flag = confidence < 0.85 or len(anomalies) > 0
            
            # Extraction
            extracted = extract_mortgage_entities(ocr_text, ai_label)
            
            page_records.append({
                "page_index": i,
                "s3_path": image_filename,
                "ai_label": ai_label,
                "confidence_score": confidence,
                "is_flagged": should_flag,
                "anomaly_flags": json.dumps(anomalies),
                "extracted_data": json.dumps(extracted)
            })
            print(f"Processed page {page_num}: {ai_label} (Conf: {confidence}) - Flags: {anomalies}")

        # 3. Notify backend about the pages
        backend_notify_url = f"{BACKEND_URL}/api/blobs/{blob_id}/pages"
        resp = requests.post(backend_notify_url, json={"pages": page_records})
        if resp.status_code != 200:
            print(f"Warning: Failed to notify backend: {resp.text}")

        # 4. Run classification logic (mock or real)
        # ... logic from paddleOCR could be integrated here ...
        
        # 5. Update blob status to READY_FOR_REVIEW
        requests.patch(f"{BACKEND_URL}/api/blobs/{blob_id}", json={"status": "COMPLETED"})
        
        print(f"[SUCCESS] Pipeline completed for Blob {blob_id}")
        doc.close()
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
            
    except Exception as e:
        # Safely cleanup temp file if it exists
        try:
            if 'temp_pdf_path' in locals() and temp_pdf_path and os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
        except: pass

        print(f"[ERROR] Pipeline failed for {blob_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        try:
            requests.patch(f"{BACKEND_URL}/api/blobs/{blob_id}", json={"status": "FAILED"})
        except Exception as req_err:
            print(f"Could not notify backend: {req_err}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
