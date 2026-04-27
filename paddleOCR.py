# Classification module — no OCR dependency.
# Uses rapidfuzz for fuzzy keyword matching against DOCUMENT_TYPES.
from rapidfuzz import fuzz
import traceback

# document dictionary
DOCUMENT_TYPES = {
    # Existing documents with additional aliases
    "GFE": [
        "good faith estimate",
        "good faith estimate gfe",
        "hud gfe",
        "gfe form"
    ],
    "URLA": [
        "uniform residential loan application",
        "urla form",
        "1003",
        "freddie mac form 65",
        "fannie mae form 1003",
        "uniform residential loan application urla",
        "residential loan application"
    ],
    "TIA": [
        "tax information authorization",
        "tax information authorization form",
        "irs form 4506",
        "4506",
        "irs 4506",
        "form 4506",
        "tax return authorization"
    ],
    "LS": [
        "loan submission sheet",
        "loan submission sheet lss",
        "submission sheet",
        "loan transmittal sheet"
    ],
    
    # New documents added
    "DEMO": [  # Demographic Information Addendum
        "demographic information addendum",
        "demographic addendum",
        "hmda demographic information addendum",
        "uniform residential loan application demographic addendum",
        "borrower demographic information",
        "demographic information form"
    ],
    "UUTS": [  # Uniform Underwriting and Transmittal Summary
        "uniform underwriting and transmittal summary",
        "uniform underwriting transmittal summary",
        "fannie mae form 1077",
        "freddie mac form 1008",
        "form 1077",
        "form 1008",
        "underwriting and transmittal summary",
        "uw transmittal summary"
    ],
    "RTTR": [  # Request for Transcript of Tax Return
        "request for transcript of tax return",
        "irs form 4506-c",
        "form 4506-c",
        "4506-c",
        "irs 4506c",
        "request for tax return transcript",
        "tax transcript request",
        "irs tax return transcript request",
        "form 4506c"
    ],
    "URAR": [  # Uniform Residential Appraisal Report
        "uniform residential appraisal report",
        "urar",
        "fannie mae form 1004",
        "freddie mac form 70",
        "form 1004",
        "form 70",
        "uniform appraisal report",
        "residential appraisal report",
        "1004 appraisal form"
    ],
    "SA": [  # Supplemental Addendum
        "supplemental addendum",
        "multi purpose supplemental addendum",
        "supplemental appraisal addendum",
        "form 1004 supplemental addendum",
        "urar supplemental addendum",
        "appraisal supplemental addendum"
    ],
    "MCA": [  # Market Conditions Addendum to the Appraisal Report
        "market conditions addendum to the appraisal report",
        "market conditions addendum",
        "fannie mae form 1004mc",
        "form 1004mc",
        "1004mc",
        "market conditions addendum appraisal",
        "appraisal market conditions addendum",
        "market conditions addendum urar"
    ],
    "UAD_DEF": [  # UNIFORM APPRAISAL DATASET (UAD) DEFINITIONS ADDENDUM
        "uniform appraisal dataset definitions addendum",
        "uad definitions addendum",
        "uniform appraisal dataset definitions",
        "uad definitions",
        "uad addendum",
        "form 1004 uad definitions addendum",
        "uad appraisal definitions addendum"
    ],
    "E&O": [  # REAL ESTATE APPRAISERS ERRORS & OMISSIONS INSURANCE POLICY DECLARATIONS PAGE
        "real estate appraisers errors & omissions insurance policy declarations page",
        "eo insurance policy declarations page",
        "appraisers errors and omissions insurance declarations",
        "eo declarations page",
        "appraiser professional liability insurance declarations page",
        "errors and omissions insurance policy appraisers",
        "appraiser e&o policy declarations"
    ],
    "APP_LICENSE": [  # REAL ESTATE APPRAISER LICENSE
        "real estate appraiser license",
        "appraiser license",
        "appraisal license",
        "state appraiser license",
        "appraiser certification",
        "certified real estate appraiser license",
        "professional appraiser license",
        "appraiser license certificate",
        "state certified appraiser license"
    ],
    "COA": [  # Certificate of Appraiser Independence
        "certificate of appraiser independence",
        "appraiser independence certificate",
        "coa",
        "certificate of independence appraiser",
        "appraiser independence certification",
        "air certificate",
        "appraiser independence requirements certificate",
        "certificate of appraiser independence form"
    ]
}


def classify_page(text_blocks):
    """
    Classifies a page based on its OCR text blocks using fuzzy keyword matching.

    Args:
        text_blocks: list of dicts with a 'text' key.

    Returns:
        tuple: (label: str, confidence: float)
            - label is a key from DOCUMENT_TYPES, or "UNCLASSIFIED"
            - confidence is a float between 0.0 and 1.0
    """
    full_text = " ".join([block['text'] for block in text_blocks]).lower()

    best_match = "UNCLASSIFIED"
    best_score = 0

    for doc_type, keywords in DOCUMENT_TYPES.items():
        for keyword in keywords:
            score = fuzz.partial_ratio(keyword, full_text)

            if score > best_score:
                best_score = score
                best_match = doc_type

    # Threshold check: score must exceed 80 to be considered a match
    if best_score > 80:
        return best_match, round(best_score / 100, 2)
    else:
        return "UNCLASSIFIED", round(best_score / 100, 2)




def split_pdf_by_segments(pdf_path, segments, output_dir="output_docs"):
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)

    for i, segment in enumerate(segments):
        new_doc = fitz.open()

        for page_num in range(segment["start_page"] - 1, segment["end_page"]):
            new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)

        output_path = os.path.join(
            output_dir,
            f"{segment['type']}_{i+1}.pdf"
        )

        new_doc.save(output_path)
        new_doc.close()

        print(f"Saved: {output_path}")

    doc.close()


def process_pdf_with_ocr(pdf_path, output_file, ocr):
    print(f"\n{'='*60}")
    print(f"Processing PDF: {pdf_path}")
    print(f"{'='*60}\n")
    

    print("Opening PDF...")
    doc = fitz.open(pdf_path)
    zoom = 200 / 72 
    mat = fitz.Matrix(zoom, zoom)

    print(f"Total pages: {len(doc)}\n")

    
    # Run OCR
    print("Running OCR on each page...")
    print("-" * 40)
    
    all_results = []

    document_segments = []
    current_doc = None
    
    
    for page_num in range(len(doc)):
        print(f"Page {page_num + 1}:")
        page = doc[page_num]
        pix = page.get_pixmap(matrix=mat)

        img = Image.open(io.BytesIO(pix.tobytes("png")))
        img_np = np.array(img)

        result = ocr.predict(img_np)
        
        page_text = []
        
        if result and len(result) > 0:
            for line in result[0]:
                text = line[1][0]
                confidence = line[1][1]
                page_text.append({'text': text, 'confidence': confidence})
                print(f"  [{confidence:.2%}] {text}")
        else:
            print("  No text found on this page")
        
        all_results.append({'page_num': page_num + 1, 'text_blocks': page_text})
        print()
        
    doc.close()

    for page_result in all_results:
        page_num = page_result['page_num']
        text_blocks = page_result['text_blocks']

        # classify_page returns (label, confidence)
        label, confidence = classify_page(text_blocks)

        if current_doc is None:
            current_doc = {
                "type": label,
                "start_page": page_num,
                "end_page": page_num
            }
        else:
            # Handle UNCLASSIFIED pages by continuing previous document
            if label == current_doc["type"] or label == "UNCLASSIFIED":
                current_doc["end_page"] = page_num
            else:
                document_segments.append(current_doc)
                current_doc = {
                    "type": label,
                    "start_page": page_num,
                    "end_page": page_num
                }

    # Append last document
    if current_doc:
        document_segments.append(current_doc)

    # Print detected segments
    print("\nDetected Document Segments:")
    for seg in document_segments:
        print(seg)
    
    # Save results
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"OCR Results for: {pdf_path}\n")
        f.write(f"{'='*60}\n\n")
        
        for page_result in all_results:
            page_num = page_result['page_num']
            text_blocks = page_result['text_blocks']
            
            f.write(f"\n{'='*60}\n")
            f.write(f"PAGE {page_num}\n")
            f.write(f"{'='*60}\n\n")
            
            if not text_blocks:
                f.write("[No text found on this page]\n\n")
                continue
            
            for block in text_blocks:
                f.write(f"[Confidence: {block['confidence']:.2%}]\n")
                f.write(f"{block['text']}\n\n")
    
    print(f"Results saved successfully!")
    print(f"\nTotal pages processed: {len(all_results)}")
    
    split_pdf_by_segments(pdf_path, document_segments)

    return all_results

if __name__ == "__main__":
    # PATH to PDF file
    PDF_PATH = r"C:\Users\alisu\OneDrive\Documents\Projects\IDP\Sample PDFs\Merged all Docs.pdf"
    OUTPUT_FILE = "ocr_results2.txt"
    USE_GPU = False  # Set to True only if have an NVIDIA GPU
    LANGUAGE = 'en'  # 'en' for English
    
    try:
        results = process_pdf_with_ocr(
            pdf_path=PDF_PATH,
            output_file=OUTPUT_FILE,
            ocr=ocr
        )
        print(f"\n✅ Done! Check '{OUTPUT_FILE}' for the extracted text.")
    except FileNotFoundError:
        print(f"\n❌ Error: Could not find the PDF file at '{PDF_PATH}'")
        print("Please update the PDF_PATH variable with the correct file path.")
    except Exception as e:
        print("❌ An error occurred:")
        traceback.print_exc()