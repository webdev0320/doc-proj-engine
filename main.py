import os
# STABILITY FLAGS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
# Disable Paddle specific flags if not using PaddleOCR library
# os.environ['FLAGS_use_mkldnn'] = '0'
# os.environ['PADDLE_ONEDNN'] = 'OFF'

import shutil
import requests
import cv2
import numpy as np
import json
import fitz
import paramiko
import boto3
import re
from io import BytesIO
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
# import easyocr  # Moved to lazy loading
from paddleOCR import classify_page
from crypto_utils import decrypt_file
from dotenv import load_dotenv

load_dotenv()

# Lazy-loaded EasyOCR reader
_reader = None

def get_reader():
    global _reader
    if _reader is None:
        print(">>> [OCR] Loading EasyOCR model (this may take a moment)...")
        import easyocr
        _reader = easyocr.Reader(['en'], gpu=False)
        print(">>> [OCR] EasyOCR ready.")
    return _reader

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

class AppendRequest(BaseModel):
    blob_id: str
    storage_path: str
    page_offset: int = 0  # Starting page index to avoid collisions with existing pages
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
                # Try downloading full path first (e.g. for Add Pages where remote has UUID)
                print(f"[DEBUG] File missing locally. Trying remote download: {req.storage_path}")
                try:
                    download_from_remote(req.storage_path, pdf_path, req.storage_settings)
                except Exception as e1:
                    # Fallback: strip UUID (first 37 chars) for SFTP Poller flow
                    if len(req.storage_path) > 37 and req.storage_path[36] == '-':
                        stripped = req.storage_path[37:]
                        print(f"[DEBUG] Full path failed, trying stripped: {stripped}")
                        download_from_remote(stripped, pdf_path, req.storage_settings)
                    else:
                        raise e1
            except Exception as e:
                print(f"[ERROR] Remote download failed: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to download file from remote storage: {e}")
        else:
            raise HTTPException(status_code=404, detail=f"PDF not found locally and no storage settings provided.")

    # Run OCR in background
    background_tasks.add_task(run_pipeline, req.blob_id, pdf_path, req.storage_settings)
    
    return {"success": True, "message": "OCR processing started in background"}


@app.post("/process-append")
async def process_append(req: AppendRequest, background_tasks: BackgroundTasks):
    """
    Appends new pages from a new file to an existing blob.
    Page indices start from req.page_offset so they don't collide with existing pages.
    """
    pdf_path = os.path.join(BLOBS_DIR, req.storage_path)
    print(f"[DEBUG] Checking local path: {pdf_path}")

    if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) == 0:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        if req.storage_settings:
            try:
                print(f"[APPEND] File missing locally. Trying remote download: {req.storage_path}")
                try:
                    download_from_remote(req.storage_path, pdf_path, req.storage_settings)
                except Exception as e1:
                    if len(req.storage_path) > 37 and req.storage_path[36] == '-':
                        stripped = req.storage_path[37:]
                        print(f"[APPEND] Full path failed, trying stripped: {stripped}")
                        download_from_remote(stripped, pdf_path, req.storage_settings)
                    else:
                        raise e1
            except Exception as e:
                print(f"[APPEND ERROR] Remote download failed: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to download append file: {e}")
        else:
            raise HTTPException(status_code=404, detail="Append file not found and no storage settings provided.")

    background_tasks.add_task(run_pipeline_append, req.blob_id, pdf_path, req.page_offset, req.storage_settings)
    return {"success": True, "message": "Append processing started in background"}



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

def _extract_field(text, pattern, default=None, multiline=False):
    """Helper: extract a single captured group from text using regex."""
    flags = re.IGNORECASE | (re.DOTALL if multiline else 0)
    m = re.search(pattern, text, flags)
    if m:
        val = m.group(1).strip()
        # Clean up common OCR noise - extra whitespace, leading colons
        val = re.sub(r'\s+', ' ', val).strip(':').strip()
        return val if val else default
    return default

# Labels that bound GFE field values — stop extracting when we see one of these
_GFE_LABELS = (
    r"(?:Name\s+of\s+Originator|Borrower|Originator\s+Address|Property\s+Address|"
    r"Originator\s+Phone|Originator\s+Email|Date\s+of\s+GFE|Purpose|Loan\s+Terms|"
    r"Initial\s+loan|Summary\s+of\s+your|Shopping\s+chart|Understanding)"
)

def _extract_gfe(text):
    """Extract fields from Good Faith Estimate (GFE) document."""
    data = {}
    # Each field: value ends at a newline OR at the next known GFE label
    boundary = r"(?=\n" + _GFE_LABELS + r"|$)"
    fields = {
        "Name of Originator": r"Name\s+of\s+Originator\s*\n([^\n]+)" + boundary,
        "Borrower":           r"\bBorrower\b\s*\n([^\n]+)" + boundary,
        "Originator Address": r"Originator\s+Address\s*\n([^\n]+)" + boundary,
        "Property Address":   r"Property\s+Address\s*\n([^\n]+)" + boundary,
        "Originator Phone Number": r"Originator\s+Phone\s*(?:Number)?\s*\n([^\n]+)",
        "Originator Email":   r"Originator\s+Email\s*\n([^\n]+)",
        "Date of GFE":        r"Date\s+of\s+GFE\s*\n([^\n]+)",
        "Loan Amount":        r"Initial\s+loan\s+amount\s+\$\s*([\d,\.]+)",
        "Interest Rate":      r"Initial\s+interest\s+rate\s+([\d\.]+\s*%)",
        "Loan Term":          r"Loan\s+[Tt]erm\s+([\d]+\s+(?:years?|months?))",
        "Loan Purpose":       r"Loan\s+Purpose\s+([^\n]+)",
        "Property Value":     r"(?:Purchase\s+price|estimated\s+to\s+be\s+worth)\s+\$\s*([\d,\.]+)",
        "Settlement Charges": r"Total\s+Settlement\s+Charges\s+\$\s*([\d,\.]+)",
        "Our Origination Charge": r"Our\s+origination\s+charge\s+\$\s*([\d,\.]+)",
        "Can Interest Rate Rise": r"Can\s+your\s+interest\s+rate\s+rise\?\s*([^\n]+)",
        "Can Loan Balance Rise":  r"Can\s+your\s+loan\s+balance\s+rise\?\s*([^\n]+)",
        "Can Monthly Amount Rise": r"Can\s+your\s+monthly\s+amount\s+owed\s+rise\?\s*([^\n]+)",
    }
    for key, pattern in fields.items():
        val = _extract_field(text, pattern)
        if val:
            data[key] = val
    return data

def _extract_urla(text):
    """Extract fields from Uniform Residential Loan Application (URLA / 1003)."""
    data = {}
    fields = {
        "Borrower Name": r"(?:Borrower's\s+Name|Borrower\s+Name)\s*:?\s*([A-Za-z\s\.\-]+?)(?=\n|Co-Borrower|Social)",
        "Co-Borrower Name": r"Co-Borrower(?:'s)?\s+Name\s*:?\s*([A-Za-z\s\.\-]+?)(?=\n|Social|Address)",
        "Property Address": r"(?:Subject\s+Property\s+Address|Property\s+Street\s+Address)\s*:?\s*([^\n]+(?:\n[^\n]{0,60})?)",
        "City": r"(?:City|City\s*,\s*State)\s*:?\s*([A-Za-z\s]+?)(?=,|\n|State|County|Zip)",
        "State": r"State\s*:?\s*([A-Z]{2})\b",
        "Zip Code": r"Zip\s*(?:Code)?\s*:?\s*(\d{5}(?:-\d{4})?)",
        "Loan Amount": r"(?:Amount\s+of\s+Loan|Loan\s+Amount)\s*\$?\s*([\d,\.]+)",
        "Interest Rate": r"Interest\s+Rate\s*:?\s*([\d\.]+\s*%)",
        "Loan Term": r"(?:No\.\s*of\s*Months|Loan\s+Term)\s*:?\s*([\d]+)",
        "Amortization Type": r"Amortization\s+Type\s*:?\s*([^\n]+)",
        "Loan Purpose": r"Purpose\s+of\s+Loan\s*:?\s*([^\n]+)",
        "Property Type": r"Property\s+will\s+be\s*:?\s*([^\n]+)",
        "Social Security Number": r"(?:Social\s+Security\s+Number|SSN)\s*[:#]?\s*([\dX\*\-]{9,11})",
        "Date of Birth": r"(?:Date\s+of\s+Birth|DOB)\s*:?\s*([\d]{1,2}[\/\-][\d]{1,2}[\/\-][\d]{2,4})",
        "Employer Name": r"(?:Name\s+and\s+Address\s+of\s+Employer|Employer\s*Name)\s*:?\s*([^\n]+)",
        "Monthly Income": r"(?:Base\s+Empl\.\s+Income|Monthly\s+Income)\s*\$?\s*([\d,\.]+)",
    }
    for key, pattern in fields.items():
        val = _extract_field(text, pattern)
        if val:
            data[key] = val
    return data

def _extract_tia(text):
    """Extract fields from Tax Information Authorization (TIA / IRS Form 8821 / 4506)."""
    data = {}
    fields = {
        "Taxpayer Name": r"(?:Taxpayer\s+name\(s\)|1\.\s+Taxpayer\s+information)\s*:?\s*([^\n]+)",
        "Taxpayer Identification Number": r"(?:Taxpayer\s+identification\s+number|TIN|SSN)\s*:?\s*([\dX\*\-]{7,11})",
        "Current Address": r"(?:Current\s+address|Street\s+address)\s*:?\s*([^\n]+)",
        "City State Zip": r"(?:City\s+or\s+town,\s+state|City,\s+state)\s*:?\s*([^\n]+)",
        "Appointee Name": r"(?:2\.\s+Appointee|Appointee\s*'?s?\s+name)\s*:?\s*([^\n]+)",
        "Appointee Address": r"(?:Appointee\s+address|CAF\s+No)\s*:?\s*([^\n]+)",
        "Tax Form Number": r"(?:Tax\s+[Ff]orm\s+[Nn]umber|Form\s+number)\s*:?\s*([^\n]+)",
        "Year or Period": r"(?:Year\(s\)\s+or\s+period\(s\)|Tax\s+year|Tax\s+period)\s*:?\s*([^\n]+)",
        "Specific Use": r"(?:4\.\s+Specific\s+use|Specific\s+use\s+not\s+recorded)\s*:?\s*([^\n]+)",
        "Signature Date": r"Signature\s*:?\s*Date\s*:?\s*([\d\/\-\.]+)",
    }
    for key, pattern in fields.items():
        val = _extract_field(text, pattern)
        if val:
            data[key] = val
    return data

def _extract_ls(text):
    """Extract fields from Loan Submission Sheet (LS)."""
    data = {}
    fields = {
        "Submitting Company": r"(?:Submitting\s+Broker[\/\-]?Lender|Company)\s*\n([^\n]+)",
        "Broker/Lender Address": r"(?:Submitting\s+Broker[\/\-]?Lender|Company)\s*\n[^\n]+\n(?:Address)\s*\n([^\n]+)",
        "Processor": r"Processor\s*\n([^\n]+)",
        "Phone Number": r"Phone\s*#\s*\n([^\n]+)",
        "Fax Number": r"Fax\s*#\s*\n([^\n]+)",
        "Estimated Close of Escrow": r"Estimated\s+Close\s+of\s+Escrow\s*\n([^\n]+)",
        "Borrower": r"Borrower\s*\n([^\n]+)",
        "Co-Borrower": r"Co-Borrower\s*\n([^\n]+)",
        "Property Address": r"Property\s+Address\s*\n([^\n]+)",
        "Program": r"Program\s*\n([^\n]+)",
        "Property Type": r"Property\s+Type\s*\n([^\n]+)",
        "Loan Amount": r"Loan\s+Amount\s*\$?\s*([\d,\.]+)",
        "Sales Price": r"Sales\s+Price\s*\$?\s*([\d,\.]+)",
        "Interest Rate": r"Interest\s+Rate\s*:?\s*([\d\.]+\s*%?)",
        "LTV": r"\bLTV\b\s*:?\s*([\d\.]+\s*%?)",
        "Amortization": r"Amortization\s*:?\s*([\d]+)",
        "Rate Lock": r"Rate\s+Lock\s*:?\s*([^\n]+)",
        "Origination Fee": r"Origination\s+Fee\s*[%\$]?\s*([^\n]+)",
        "Appraisal Fee": r"Appraisal\s+Fee\s*:?\s*\$?\s*([\d,\.]+)",
        "Escrow Company": r"Escrow\s+Company[\/\-]?Attorney\s*\nCompany\s*\n([^\n]+)",
        "Title Company": r"Title\s+Company\s*\nCompany\s*\n([^\n]+)",
        "Appraiser": r"Appraiser\s*\n([^\n]+)",
    }
    for key, pattern in fields.items():
        val = _extract_field(text, pattern)
        if val:
            data[key] = val
    return data

def _extract_all_labels_values(ocr_text):
    """
    Universal sweep: extract ALL label/value pairs from OCR text.
    Detects two common patterns in scanned forms:
      1. "Label: Value" on the same line
      2. "Label\nValue" where the label is on one line and value on the next
    Skips noise lines (too short, all-caps boilerplate, page numbers, etc.)
    """
    data = {}
    if not ocr_text:
        return data

    lines = ocr_text.split('\n')

    # Noise filters
    _SKIP_WORDS = {
        '', 'page', 'of', 'yes', 'no', 'true', 'false', 'n/a', 'none',
        'the', 'a', 'an', 'and', 'or', 'for', 'to', 'in', 'on', 'at',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'if', 'not',
    }

    def _is_noise(s):
        """Check if a string is noise (too short, just numbers, common word)."""
        s = s.strip()
        if not s or len(s) < 2 or len(s) > 80:
            return True
        if re.match(r'^[\d\s\.\,\-\$\%\/]+$', s):
            return True
        if s.lower().strip(':').strip() in _SKIP_WORDS:
            return True
        # Skip long paragraph-like text
        if len(s) > 60 and s.count(' ') > 10:
            return True
        return False

    def _looks_like_label(s):
        """Check if a line looks like a form field label (short, Title/UPPER case, no $)."""
        s = s.strip()
        if not s or len(s) < 2 or len(s) > 50:
            return False
        if '$' in s or '%' in s:
            return False
        if re.match(r'^[\d\s\.\,\-\$\%\/]+$', s):
            return False
        if s[0].isupper() and s.count(' ') <= 6:
            return True
        return False

    def _clean_label(s):
        """Normalize a label string."""
        s = s.strip().rstrip(':').strip()
        return s

    # Pattern 1: "Label: Value" or "Label   Value" on the same line
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Match "Label: Value"
        m1 = re.match(r'^([A-Za-z][A-Za-z\s\.\-\/\(\)#]{1,50}):\s+(.+)$', line)
        if m1:
            label = _clean_label(m1.group(1))
            value = m1.group(2).strip()
            if not _is_noise(label) and value and label not in data:
                data[label] = value
            continue
            
        # Match "Label   Value" (separated by 2 or more spaces/tabs)
        m2 = re.match(r'^([A-Za-z][A-Za-z\s\.\-\/\(\)#]{1,50})(?:\s{2,}|\t+)(.+)$', line)
        if m2 and not ':' in line: # avoid clashing with Pattern 1
            label = _clean_label(m2.group(1))
            value = m2.group(2).strip()
            if _looks_like_label(label) and not _is_noise(label) and value and label not in data:
                data[label] = value

    # Pattern 2: "Label\nValue" — label on one line, value on the next
    for i in range(len(lines) - 1):
        label_line = lines[i].strip()
        value_line = lines[i + 1].strip()

        if not label_line or not value_line:
            continue

        # Label must look like a form field name
        if not _looks_like_label(label_line):
            continue

        # If it has a colon, it must be at the end.
        if ':' in label_line and not label_line.endswith(':'):
            continue

        # Avoid chaining (e.g., "Name of Originator\nBorrower\nJohn Doe")
        # If value_line also looks like a label, check if it's acting as a label for the NEXT line
        if _looks_like_label(value_line) and not re.search(r'[\d@\$%#\(\)]', value_line):
            is_definite_label = label_line.endswith(':')
            if not is_definite_label:
                # Look ahead to see if value_line is actually a label for lines[i+2]
                if i + 2 < len(lines):
                    next_val = lines[i + 2].strip()
                    if next_val and not _looks_like_label(next_val):
                        # value_line is acting as a label for next_val
                        continue
                else:
                    # End of file, ambiguous but likely not chaining
                    pass

        label = _clean_label(label_line)
        if not _is_noise(label) and value_line and label not in data:
            data[label] = value_line

    return data


def extract_entities(ocr_text, doc_type):
    """Extract structured fields from OCR text based on the document type.
    
    Uses two strategies:
    1. Document-type-specific regex patterns for high-priority known fields
    2. Universal label/value sweep to capture ALL remaining fields
    """
    data = {}
    if not isinstance(ocr_text, str) or not ocr_text:
        return data

    upper_doc = (doc_type or "").upper()

    # Step 1: Run the universal sweep FIRST to get all label/value pairs
    data = _extract_all_labels_values(ocr_text)

    # Step 2: Run doc-type-specific extractor and MERGE (overrides universal)
    typed_data = {}
    if upper_doc == "GFE":
        typed_data = _extract_gfe(ocr_text)
    elif upper_doc == "URLA":
        typed_data = _extract_urla(ocr_text)
    elif upper_doc in ("TIA", "RTTR"):
        typed_data = _extract_tia(ocr_text)
    elif upper_doc == "LS":
        typed_data = _extract_ls(ocr_text)

    # Type-specific fields override universal ones (they are more accurate)
    data.update(typed_data)

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

        # Decrypt file before opening
        decrypted_path = pdf_path + ".dec"
        decrypt_file(pdf_path, decrypted_path)
        
        if not os.path.exists(decrypted_path) or os.path.getsize(decrypted_path) == 0:
            print(f"!!! [PIPELINE ERROR] Decryption failed or resulted in empty file: {decrypted_path}")
            return

        print(f">>> [PIPELINE] Opening PDF: {decrypted_path}")
        doc = fitz.open(decrypted_path)
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
                    # Force leading slash if needed, or try both
                    remote_image_path = f"pages/{image_filename}"
                    print(f">>> [PIPELINE] Uploading page {page_num} to SFTP: {remote_image_path}")
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
                    ocr_reader = get_reader()
                    res = ocr_reader.readtext(image_path)
                    ocr_text = "\n".join([item[1] for item in res]) if res else ""
            except:
                ocr_text, res = "", None

            text_blocks = [{'text': item[1], 'confidence': item[2]} for item in res] if (res and res != "NativeTextExtractor") else [{'text': ocr_text, 'confidence': 1.0}]
            ai_label, fuzzy_confidence = classify_page(text_blocks)
            confidence = fuzzy_confidence if res else max(fuzzy_confidence - 0.1, 0.0)
            should_flag = confidence < 0.85 or len(anomalies) > 0
            extracted = extract_entities(ocr_text, ai_label)
            
            page_records.append({
                "page_index": i, "s3_path": image_filename, "ai_label": ai_label,
                "confidence_score": confidence, "is_flagged": should_flag,
                "anomaly_flags": json.dumps(anomalies), "extracted_data": json.dumps(extracted)
            })
            print(f"Page {page_num}: {ai_label} (Conf: {confidence})")

        try:
            r = requests.post(f"{BACKEND_URL}/api/blobs/{blob_id}/pages", json={"pages": page_records}, timeout=30)
            print(f">>> [CALLBACK] POST /api/blobs/{blob_id}/pages -> {r.status_code} {r.text}")
        except Exception as cb_e:
            print(f">>> [CALLBACK ERROR] Failed to POST pages to backend: {cb_e}")

        try:
            r2 = requests.patch(f"{BACKEND_URL}/api/blobs/{blob_id}", json={"status": "COMPLETED"}, timeout=10)
            print(f">>> [CALLBACK] PATCH /api/blobs/{blob_id} -> {r2.status_code} {r2.text}")
        except Exception as cb_e:
            print(f">>> [CALLBACK ERROR] Failed to PATCH backend status: {cb_e}")
        doc.close()
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[ERROR] Pipeline failed: {e}\n{tb}")
        try:
            requests.patch(
                f"{BACKEND_URL}/api/blobs/{blob_id}",
                json={"status": "FAILED", "error": str(e), "trace": tb}
            )
        except Exception as notify_err:
            print(f"[ERROR] Failed to notify backend of failure: {notify_err}")

def run_pipeline_append(blob_id: str, pdf_path: str, page_offset: int, storage_settings: Optional[Dict[str, Any]] = None):
    """Processes a new file and appends its pages to an existing blob, starting from page_offset."""
    try:
        print(f">>> [APPEND] Starting for Blob: {blob_id}, offset: {page_offset}")
        if not os.path.exists(pdf_path):
            print(f"!!! [APPEND ERROR] File does not exist: {pdf_path}")
            return

        # Decrypt file before opening
        decrypted_path = pdf_path + ".dec"
        decrypt_file(pdf_path, decrypted_path)

        doc = fitz.open(decrypted_path)
        print(f">>> [APPEND] PDF opened. New pages: {len(doc)}")
        page_records = []

        for i in range(len(doc)):
            abs_index = page_offset + i          # absolute index in the blob
            page_num = abs_index + 1             # 1-based, unique per blob

            page = doc[i]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            image_filename = f"{blob_id}_{page_num}.png"
            image_path = os.path.join(PAGES_DIR, image_filename)
            pix.save(image_path)

            if storage_settings:
                try:
                    upload_to_remote(image_path, f"pages/{image_filename}", storage_settings)
                except Exception as ue:
                    print(f"!!! [APPEND] Page upload failed: {ue}")

            anomalies, blur_score, skew = scan_health_check(image_path)
            pdf_text = page.get_text("text").strip()

            try:
                if len(pdf_text) > 20:
                    ocr_text = pdf_text
                    res = "NativeTextExtractor"
                else:
                    ocr_reader = get_reader()
                    res = ocr_reader.readtext(image_path)
                    ocr_text = "\n".join([item[1] for item in res]) if res else ""
            except:
                ocr_text, res = "", None

            text_blocks = [{'text': item[1], 'confidence': item[2]} for item in res] if (res and res != "NativeTextExtractor") else [{'text': ocr_text, 'confidence': 1.0}]
            ai_label, fuzzy_confidence = classify_page(text_blocks)
            confidence = fuzzy_confidence if res else max(fuzzy_confidence - 0.1, 0.0)
            should_flag = confidence < 0.85 or len(anomalies) > 0
            extracted = extract_entities(ocr_text, ai_label)

            page_records.append({
                "page_index": abs_index,
                "s3_path": image_filename,
                "ai_label": ai_label,
                "confidence_score": confidence,
                "is_flagged": should_flag,
                "anomaly_flags": json.dumps(anomalies),
                "extracted_data": json.dumps(extracted)
            })
            print(f"[APPEND] Page {abs_index}: {ai_label} (Conf: {confidence})")

        # Post back with mode=append so existing pages are NOT deleted
        try:
            r = requests.post(
                f"{BACKEND_URL}/api/blobs/{blob_id}/pages",
                json={"pages": page_records, "mode": "append"},
                timeout=30
            )
            print(f">>> [CALLBACK] POST /api/blobs/{blob_id}/pages (append) -> {r.status_code} {r.text}")
        except Exception as cb_e:
            print(f">>> [CALLBACK ERROR] Failed to POST append pages to backend: {cb_e}")

        try:
            r2 = requests.patch(f"{BACKEND_URL}/api/blobs/{blob_id}", json={"status": "COMPLETED"}, timeout=10)
            print(f">>> [CALLBACK] PATCH /api/blobs/{blob_id} -> {r2.status_code} {r2.text}")
        except Exception as cb_e:
            print(f">>> [CALLBACK ERROR] Failed to PATCH backend status: {cb_e}")
        doc.close()
        print(f">>> [APPEND] Done. Added {len(page_records)} pages to blob {blob_id}")
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[APPEND ERROR] Pipeline failed: {e}\n{tb}")
        try:
            requests.patch(
                f"{BACKEND_URL}/api/blobs/{blob_id}",
                json={"status": "FAILED", "error": str(e), "trace": tb}
            )
        except Exception as notify_err:
            print(f"[APPEND ERROR] Failed to notify backend of failure: {notify_err}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
