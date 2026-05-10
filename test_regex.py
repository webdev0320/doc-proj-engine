import re

def extract_entities(ocr_text):
    data = {}
    text = ocr_text
    
    patterns = {
        "Registration No.": r"(?i)Registration\s*No\.?[\s:]*([A-Z0-9-]+)",
        "Date of Registration": r"(?i)Date\s*of\s*Registration[\s:]*([\d]{1,2}-[A-Za-z]{3}-[\d]{4}|[\d]{1,2}/[\d]{1,2}/[\d]{4})",
        "Type of Person": r"(?i)Type\s*of\s*Person[\s:]*([A-Za-z\s]+?)(?=\n|Name|Address|$)",
        "Name": r"(?i)\bName[\s:]*([A-Za-z\s\.\-]+?)(?=\n|Address|Tax Office|Type of Person|$)",
        "Address": r"(?i)Address[\s:]*(.+?)(?=\n|Tax Office|Activity Type|$)",
        "Tax Office": r"(?i)Tax\s*Office[\s:]*(.+?)(?=\n|Activity Type|$)",
        "Activity Type": r"(?i)Activity\s*Type[\s:]*(.+?)(?=\n|$)"
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            val = match.group(1).strip()
            if val:
                data[key] = val
    return data

text = """Registration No. 3740584191741
Date of Registration 27-Dec-2022
Type of Person Individual
Name MUHAMMAD ARBAZ
Address 5, 16/ 19 / 1, New Gulzar-e-quaid, Rawalpindi, Rawalpindi.
Tax Office RTO RAWALPINDI
Activity Type Other"""

print("Test 1:", extract_entities(text))

text2 = """Registration No.
3740584191741
Date of Registration
27-Dec-2022
Type of Person
Individual
Name
MUHAMMAD ARBAZ
Address
5, 16/ 19 / 1, New Gulzar-e-quaid, Rawalpindi, Rawalpindi.
Tax Office
RTO RAWALPINDI
Activity Type
Other"""

print("Test 2:", extract_entities(text2))
