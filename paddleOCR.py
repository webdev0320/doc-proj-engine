# Classification module — no OCR dependency.
# Uses rapidfuzz for fuzzy keyword matching against DOCUMENT_TYPES.
from rapidfuzz import fuzz
import re
import traceback

# document dictionary
DOCUMENT_TYPES = {
    "GFE": [
        "good faith estimate",
        "good faith estimate gfe",
        "hud-1 gfe",
        "gfe form hud"
    ],
    "4506": [
        "tax information authorization",
        "tax information authorization form",
        "irs form 4506",
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
    "BANK_STATEMENT": [
        "bank statement",
        "bank statements",
        "statement of account",
        "account statement",
        "bank account statement",
        "account summary",
        "transaction history",
        "available balance",
        "statement period",
        "beginning balance",
        "ending balance"
    ],
    "CREDIT_AUTHORIZATION": [
        "credit authorization form",
        "credit authorization",
        "authorization to obtain credit",
        "credit inquiry authorization"
    ],
    "BORROWER_CERTIFICATION_AUTHORIZATION": [
        "borrower's certification authorization",
        "borrower's certification & authorization",
        "borrower certification authorization",
        "borrower certification & authorization",
        "borrower certification form"
    ],
    "BORROWER_CONSENT_TAX_RETURN": [
        "borrower consent to tax return",
        "borrower consent to the use of tax return information",
        "borrower consent to tax returns",
        "consent to tax return",
        "tax return consent"
    ],
    "BORROWER_APPLICATION_CERTIFICATION": [
        "borrower application certification",
        "borrower application certification form",
        "borrower application certification statement"
    ],
    "ESCROW_INSTRUCTIONS": [
        "escrow instructions",
        "escrow closing instructions",
        "preliminary change of ownership report",
        "escrow instructions for",
        "escrow instructions and closing",
        "closing instructions",
        "escrow company instructions",
        "title and escrow instructions",
        "escrow direction letter"
    ],
    "TITLE_POLICY": [
        "title insurance policy",
        "title commitment",
        "title policy document",
        "title report"
    ],
    "MAVENT_REPORT": [
        "mavent report",
        "mavent compliance",
        "compliance analysis report",
        "trid compliance",
        "respa compliance",
        "tila compliance",
        "regulation z compliance",
        "hmda compliance report",
        "ability to repay",
        "ability-to-repay",
        "qm atr analysis",
        "closing disclosure compliance",
        "loan estimate compliance",
        "mavent analsis",
        "mavent review"
    ],
    "CREDIT_REPORT": [
        "consumer credit report",
        "credit history report",
        "transunion",
        "equifax",
        "experian",
        "fico score",
        "credit score",
        "trade line",
        "account history",
        "payment history",
        "credit inquiry",
        "inquiry history",
        "public records",
        "credit summary",
        "revolving account",
        "installment account"
    ],
    "DRIVER_LICENSE": [
        "driver licence",
        "driver license",
        "driver's license",
        "driver identification"
    ],
    "DEMO": [
        "demographic information addendum",
        "demographic addendum",
        "hmda demographic information addendum",
        "uniform residential loan application demographic addendum",
        "borrower demographic information",
        "demographic information form"
    ],
    "UUTS": [
        "uniform underwriting and transmittal summary",
        "uniform underwriting transmittal summary",
        "fannie mae form 1077",
        "freddie mac form 1008",
        "underwriting and transmittal summary",
        "uw transmittal summary"
    ],
    "RTTR": [
        "request for transcript of tax return",
        "irs form 4506-c",
        "form 4506-c",
        "irs 4506c",
        "request for tax return transcript",
        "tax transcript request",
        "irs tax return transcript request",
        "form 4506c"
    ],
    "URAR": [
        "uniform residential appraisal report",
        "fannie mae form 1004",
        "freddie mac form 70",
        "uniform appraisal report",
        "residential appraisal report",
        "1004 appraisal form"
    ],
    "SA": [
        "supplemental addendum",
        "multi purpose supplemental addendum",
        "supplemental appraisal addendum",
        "form 1004 supplemental addendum",
        "appraisal supplemental addendum"
    ],
    "MCA": [
        "market conditions addendum to the appraisal report",
        "market conditions addendum",
        "fannie mae form 1004mc",
        "form 1004mc",
        "1004mc",
        "market conditions addendum appraisal",
        "appraisal market conditions addendum"
    ],
    "UAD_DEF": [
        "uniform appraisal dataset definitions addendum",
        "uad definitions addendum",
        "uniform appraisal dataset definitions",
        "uad definitions",
        "uad addendum",
        "form 1004 uad definitions addendum",
        "uad appraisal definitions addendum"
    ],
    "E&O": [
        "errors & omissions insurance policy declarations page",
        "eo insurance policy declarations page",
        "appraisers errors and omissions insurance declarations",
        "eo declarations page",
        "appraiser professional liability insurance declarations page",
        "errors and omissions insurance policy appraisers",
        "appraiser e&o policy declarations"
    ],
    "APP_LICENSE": [
        "real estate appraiser license",
        "state appraiser license",
        "certified real estate appraiser license",
        "professional appraiser license",
        "appraiser license certificate",
        "state certified appraiser license"
    ],
    "COA": [
        "certificate of appraiser independence",
        "appraiser independence certificate",
        "appraiser independence certification",
        "appraiser independence requirements certificate",
        "certificate of appraiser independence form"
    ],
    "URLA": [
        "uniform residential loan application",
        "freddie mac form 65",
        "fannie mae form 1003",
        "uniform residential loan application urla",
        "residential loan application"
    ]
}

# Keywords that must appear as standalone words (not substrings) to avoid false positives
_STRICT_KEYWORDS = {"COA", "URLA"}

# Minimum keyword length to prevent over-fuzzy matching
_MIN_KEYWORD_LEN = 5


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
            kw_lower = keyword.lower()
            # For strict keyword types, require whole-word match to avoid false positives
            if doc_type in _STRICT_KEYWORDS and len(kw_lower) <= 4:
                # Use token_set_ratio for short keywords to avoid substring false positives
                if re.search(r'\b' + re.escape(kw_lower) + r'\b', full_text):
                    score = 95
                else:
                    continue
            elif len(kw_lower) < _MIN_KEYWORD_LEN:
                # Skip overly short keywords that cause false matches
                continue
            else:
                score = fuzz.partial_ratio(kw_lower, full_text)

            if score > best_score:
                best_score = score
                best_match = doc_type

    if best_score > 80:
        return best_match, round(best_score / 100, 2)
    else:
        return "UNCLASSIFIED", round(best_score / 100, 2)


def _detect_page_of_pattern(text):
    """Detect 'Page X of Y' or 'X/Y' patterns at bottom of page text."""
    patterns = [
        r'page\s+(\d+)\s+of\s+(\d+)',
        r'(\d+)\s*/\s*(\d+)',
        r'page\s+(\d+)\s+/\s*(\d+)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None, None


def classify_documents(pages_data):
    """
    Context-aware document classifier. Takes per-page classification results
    and applies multi-page grouping, neighbor propagation, and confidence boosting.

    Args:
        pages_data: list of dicts, each with keys:
            - page_index (int)
            - text (str): full OCR/native text for the page
            - ai_label (str): per-page classification label
            - confidence_score (float): per-page confidence

    Returns:
        list of dicts with updated 'ai_label' and 'confidence_score'
    """
    if not pages_data:
        return pages_data

    # Step 1: Per-page re-classification using text (already done before calling this)

    # Step 2: Detect multi-page document boundaries via "Page X of Y" patterns
    page_groups = {}  # group_id -> set of page indices
    group_lookup = {}  # page_index -> group_id
    group_counter = 0

    for page in pages_data:
        text = page.get('text', '')
        current, total = _detect_page_of_pattern(text)
        if current is not None and total is not None and total > 1:
            # Find if any neighbor is already in a group for this document
            idx = page['page_index']
            found_group = None
            for neighbor_idx in range(max(0, idx - total), min(len(pages_data), idx + total)):
                if neighbor_idx in group_lookup:
                    found_group = group_lookup[neighbor_idx]
                    break

            if found_group is not None:
                group_lookup[idx] = found_group
                page_groups[found_group].add(idx)
            else:
                gid = group_counter
                group_counter += 1
                page_groups[gid] = {idx}
                group_lookup[idx] = gid

    # For pages in a group, assign the type from the highest-confidence page
    for gid, members in page_groups.items():
        if len(members) < 2:
            continue
        member_pages = [p for p in pages_data if p['page_index'] in members]
        # Find the best label by confidence
        best = max(member_pages, key=lambda p: p['confidence_score'])
        best_label = best['ai_label']
        # Only override if majority isn't already the same type
        label_counts = {}
        for p in member_pages:
            label_counts[p['ai_label']] = label_counts.get(p['ai_label'], 0) + 1
        if label_counts.get(best_label, 0) >= len(member_pages) // 2:
            for p in member_pages:
                p['ai_label'] = best_label
                p['confidence_score'] = max(p['confidence_score'], 0.90)

    # Step 3: Consecutive same-type boosting
    # If 3+ consecutive pages share a type, boost confidence for that group
    i = 0
    while i < len(pages_data):
        j = i
        while j < len(pages_data) and pages_data[j]['ai_label'] == pages_data[i]['ai_label']:
            j += 1
        run_len = j - i
        if run_len >= 3 and pages_data[i]['ai_label'] != "UNCLASSIFIED":
            for k in range(i, j):
                pages_data[k]['confidence_score'] = max(pages_data[k]['confidence_score'], 0.92)
        i = j

    # Step 4: Neighbor propagation for low-confidence pages
    # If a page has low confidence and both neighbors agree on a different type, override
    for i in range(1, len(pages_data) - 1):
        if pages_data[i]['confidence_score'] >= 0.88:
            continue
        prev_label = pages_data[i - 1]['ai_label']
        next_label = pages_data[i + 1]['ai_label']
        if prev_label == next_label and prev_label != pages_data[i]['ai_label'] and prev_label != "UNCLASSIFIED":
            pages_data[i]['ai_label'] = prev_label
            pages_data[i]['confidence_score'] = 0.88

    # Step 5: Island detection — single page different from surrounding same-type pages
    for i in range(1, len(pages_data) - 1):
        if pages_data[i]['confidence_score'] >= 0.90:
            continue
        # Check if surrounded by same type
        prev_label = pages_data[i - 1]['ai_label']
        next_label = pages_data[i + 1]['ai_label'] if i + 1 < len(pages_data) else None
        curr_label = pages_data[i]['ai_label']
        if prev_label == curr_label:
            continue  # Not an island
        if prev_label == next_label and prev_label != curr_label and prev_label != "UNCLASSIFIED":
            # Check if there's more context (2+ pages of same type on each side)
            left_count = 0
            for k in range(i - 1, -1, -1):
                if pages_data[k]['ai_label'] == prev_label:
                    left_count += 1
                else:
                    break
            right_count = 0
            for k in range(i + 1, len(pages_data)):
                if pages_data[k]['ai_label'] == next_label:
                    right_count += 1
                else:
                    break
            if left_count >= 2 and right_count >= 2:
                pages_data[i]['ai_label'] = prev_label
                pages_data[i]['confidence_score'] = 0.87

    return pages_data




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