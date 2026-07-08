import os
import fitz  # PyMuPDF
from PIL import Image
import io
import json
import easyocr
import sys

# Add engine root to path to import main
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import extract_entities, classify_page

# Initialize EasyOCR
reader = easyocr.Reader(['en'], gpu=True)

PDF_PATH = r"D:\projects\doc-project\engine\files\test-1_merged_compressed.pdf"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

metadata = []

print(f"Loading PDF: {PDF_PATH}")
doc = fitz.open(PDF_PATH)

for page_num in range(len(doc)):
    print(f"--- Processing Page {page_num + 1}/{len(doc)} ---")
    page = doc.load_page(page_num)
    
    # 1. Render to Image
    pix = page.get_pixmap(dpi=150) # 150 DPI is enough for Donut
    img_data = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_data)).convert("RGB")
    
    img_filename = f"page_{page_num + 1}.png"
    img_path = os.path.join(DATA_DIR, img_filename)
    img.save(img_path)
    
    # 2. Extract OCR
    print("Running OCR...")
    ocr_results = reader.readtext(img_data)
    ocr_text = "\n".join([item[1] for item in ocr_results])
    
    # 3. Predict Doc Type (fallback to GFE if unclear to trigger max extraction)
    # classify_page expects a list of dicts: [{'text': ...}]
    text_blocks = [{'text': item[1], 'confidence': item[2]} for item in ocr_results]
    doc_type, _ = classify_page(text_blocks)
    if doc_type == "UNKNOWN":
        doc_type = "GFE" # Use GFE as fallback for universal sweep
        
    # 4. Extract Entities using our robust backend logic
    print("Extracting entities...")
    extracted_data = extract_entities(ocr_text, doc_type)
    
    print(f"\nExtracted Fields for Page {page_num + 1}:")
    print(json.dumps(extracted_data, indent=2))
    
    # 5. Format for Donut Ground Truth
    gt_parse = {"gt_parse": extracted_data}
    
    # Donut requires jsonl format: {"file_name": "...", "ground_truth": "{\"gt_parse\": {...}}"}
    metadata.append({
        "file_name": img_filename,
        "ground_truth": json.dumps(gt_parse)
    })
    print("\n")

# Save metadata.jsonl
metadata_path = os.path.join(DATA_DIR, "metadata.jsonl")
with open(metadata_path, 'w', encoding='utf-8') as f:
    for item in metadata:
        f.write(json.dumps(item) + '\n')

print(f"Dataset preparation complete! Saved to {metadata_path}")
