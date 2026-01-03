import os
import shutil
import json
import re
import time
import cv2
import numpy as np
import fitz  # PyMuPDF
import io
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

# ==========================================
# 1. CONFIGURATION
# ==========================================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Input/Output
INPUT_FOLDER = r"raw_pdfs/NEET_PYQPs"
OUTPUT_ROOT = "ROOT"

# METADATA: strictly follows SET_NAME_NUMBER convention
METADATA = {
    "set": "NEET_2025_PW",
    "grade": 12,
    "reference": "NEET_2025_PW",
    "prepmode": "NEET",
    "subject": "PCB"
}

GARBAGE_PATTERNS = [
    r"Physics\s*-\s*Section\s*[A-Z]",
    r"Chemistry\s*-\s*Section\s*[A-Z]",
    r"Biology\s*-\s*Section\s*[A-Z]",
    r"Botany\s*-\s*Section\s*[A-Z]",
    r"Zoology\s*-\s*Section\s*[A-Z]",
    r"NEET", r"NEET\s*\(UG\)",
    r"Test Booklet Code",
    r"Shift\s*\d", r"Page\s*:\s*\d+",
    r"SPACE\s*FOR\s*ROUGH\s*WORK",
    r"Rough\s*Work",
    r"PW", r"Physics\s*Wallah"
]

# Processing Constants
CROP_BUFFER = 5   # Minimal buffer to exclude logos
WATERMARK_THRESHOLD = 240 # Very high threshold to kill gray logos
RENDER_DPI = 400
API_DELAY_SECONDS = 2
MAX_RETRIES = 3

# ==========================================
# 2. HYBRID VISION ENGINE (Smart Snap & Clean)
# ==========================================

def get_fitz_rect(page, box_norm):
    """Converts Gemini 1000-scale box to PDF Point Rect."""
    h, w = page.rect.height, page.rect.width
    ymin, xmin, ymax, xmax = box_norm
    return fitz.Rect(
        (xmin / 1000) * w, 
        (ymin / 1000) * h, 
        (xmax / 1000) * w, 
        (ymax / 1000) * h
    )

def extract_raw_image_if_exists(doc, page, box_norm, save_path):
    """
    PRIORITY 1: Extract the EXACT raw image object from PDF.
    This creates the cleanest possible output (no text, perfect resolution).
    """
    try:
        search_rect = get_fitz_rect(page, box_norm)
        image_list = page.get_images(full=True)
        
        best_image = None
        max_overlap = 0

        for img in image_list:
            xref = img[0]
            img_rects = page.get_image_rects(xref)
            
            for img_rect in img_rects:
                intersect = search_rect & img_rect
                if intersect.is_empty: continue
                
                overlap_area = intersect.width * intersect.height
                search_area = search_rect.width * search_rect.height
                
                if (overlap_area / search_area) > 0.3:
                    if overlap_area > max_overlap:
                        max_overlap = overlap_area
                        best_image = xref

        if best_image:
            base_image = doc.extract_image(best_image)
            pil_img = Image.open(io.BytesIO(base_image["image"]))
            pil_img.save(save_path, format="PNG")
            return True

    except Exception:
        pass 
    return False

def process_and_save_crop(original_img, box_norm, save_path):
    """
    PRIORITY 2: Smart Crop (OpenCV).
    Snaps to ink boundaries to remove surrounding text.
    """
    if not box_norm or len(box_norm) != 4: return False

    h_img, w_img, _ = original_img.shape
    ymin, xmin, ymax, xmax = box_norm

    # 1. LOOSE CROP (Capture area + buffer)
    buffer = CROP_BUFFER
    y1 = max(0, int((ymin / 1000) * h_img) - buffer)
    x1 = max(0, int((xmin / 1000) * w_img) - buffer)
    y2 = min(h_img, int((ymax / 1000) * h_img) + buffer)
    x2 = min(w_img, int((xmax / 1000) * w_img) + buffer)
    
    loose_crop = original_img[y1:y2, x1:x2]
    if loose_crop.size == 0: return False

    # 2. SMART SNAP: Find Ink Contours
    # Convert to gray
    gray = cv2.cvtColor(loose_crop, cv2.COLOR_BGR2GRAY)
    
    # --- ADVANCED LOGO SUBTRACTION ---
    # Logos are usually grey (e.g. > 150 intensity) while ink is black (< 100).
    # Use a strict threshold to capture ONLY deep blacks (Text/Lines).
    # watermarks in these PDFs are usually around 200-240.
    _, ink_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # --- SMART CLEANING: Remove orphan noise at edges ---
    # Find contours of all ink
    cnts, _ = cv2.findContours(ink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a clean mask
    clean_mask = np.zeros_like(ink_mask)
    h_img, w_img = ink_mask.shape
    center_box = [w_img*0.1, h_img*0.1, w_img*0.9, h_img*0.9] # Keep things in central 80%
    
    if cnts:
        # Sort by area
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        
        # Keep ALL significantly large contours (not just the top 1)
        # This fixes the issue where a logo might be one contour and the figure another.
        # But since we filtered for BLACK ink above, the logo shouldn't be here!
        for c in cnts:
            if cv2.contourArea(c) > 50: # Ignore tiny specks
                 cv2.drawContours(clean_mask, [c], -1, 255, -1)
    else:
        clean_mask = ink_mask # Fallback

    # Dilate to connect broken lines on the CLEAN mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) 
    dilated_mask = cv2.dilate(clean_mask, kernel, iterations=2)
    
    coords = cv2.findNonZero(dilated_mask)
    final_crop = loose_crop
    
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        if w > 10 and h > 10:
            # Snap to the exact ink bounds + aesthetic pad
            pad = 5 
            cy1 = max(0, y - pad)
            cx1 = max(0, x - pad)
            cy2 = min(loose_crop.shape[0], y + h + pad)
            cx2 = min(loose_crop.shape[1], x + w + pad)
            final_crop = loose_crop[cy1:cy2, cx1:cx2]

    # 3. CLEAN BACKGROUND (Final Polish)
    # Re-apply threshold to the final crop to kill any remaining light grey pixels
    # We want to keep the "black" content but whitelist the background
    gray_final = cv2.cvtColor(final_crop, cv2.COLOR_BGR2GRAY)
    _, clean_mask = cv2.threshold(gray_final, 210, 255, cv2.THRESH_BINARY) # Kill light grey
    final_clean_img = np.where(clean_mask[..., None] == 255, 255, final_crop)

    cv2.imwrite(save_path, final_clean_img)
    return True

# ==========================================
# 3. MAIN EXTRACTION
# ==========================================
model = None 

class DirectExtractor:
    def __init__(self, pdf_filename, set_override=None, reference_override=None):
        self.pdf_path = os.path.join(INPUT_FOLDER, pdf_filename)
        self.doc = fitz.open(self.pdf_path)
        
        # Override Metadata if provided
        self.current_set = set_override if set_override else METADATA['set']
        self.current_ref = reference_override if reference_override else METADATA['reference']
        
        # Strict Folder Structure
        self.set_dir = os.path.join(OUTPUT_ROOT, METADATA['prepmode'], METADATA['subject'], self.current_set)
        self.quarantine_dir = os.path.join(self.set_dir, "_QUARANTINE")
        os.makedirs(self.set_dir, exist_ok=True)
        os.makedirs(self.quarantine_dir, exist_ok=True)
        
        self.final_json = []
        self.answer_key_map = {}
        print(f"[OUTPUT] {self.set_dir}")

    def clean_q_text(self, text):
        if not text: return ""
        clean = " ".join(text.strip().split())
        for p in GARBAGE_PATTERNS: clean = re.sub(p, "", clean, flags=re.IGNORECASE)
        # Handle "Q1.", "1.", "1)", "(1)" prefixes
        return re.sub(r'^(Q\s*)?\d+[\.\)\s]+', '', clean.strip()).strip()

    def get_gemini_analysis(self, pil_image):
        prompt = """
        Analyze this NEET exam page and extract ALL questions. Return ONLY valid JSON.
        
        STRICTLY IGNORE:
        - Headers, Footers, Page Numbers, "SPACE FOR ROUGH WORK"
        - Watermarks, Brand Logos (especially "PW" or "Physics Wallah"), Coaching Institute Names
        - Exam Instructions, Section separators (e.g. "Physics Section A")
        - OMR/Bubble Sheets (full page answer sheets) - DO NOT extract bubbles as questions.
        - Detailed Explanations, Solutions, Hints (we ONLY want questions and options).
        - Answer Key tables at the end of the chapter (ignore them).
        
        QUESTION NUMBER FORMATS (all valid):
        - "Q1", "q1", "Q.1", "1.", "1)", "(1)", "Question 1"
        - Extract just the NUMBER as "q_id" (e.g., "1", "45", "63")
        
        QUESTION TYPES:
        1. MCQ (Multiple Choice): Has 4 options (A/B/C/D or 1/2/3/4)
        2. ASSERTION-REASON: 4 options (Both true, etc.). Treat as MCQ.
        3. NUMERICAL/FILL-IN-BLANK: No options, answer is a number. Set "options": [] and "question_type": "numerical"
        
        OPTION FORMATS (all valid):
        - "(A)", "(B)", "(C)", "(D)" 
        - "A.", "B.", "C.", "D."
        - "(1)", "(2)", "(3)", "(4)"
        - "1.", "2.", "3.", "4."
        - "a.", "b.", "c.", "d."
        
        CRITICAL IMAGE RULES:
        1. "diagram_box" is ONLY for figures/diagrams in the QUESTION TEXT itself.
        2. If OPTIONS contain graphs/figures (common in Biology/Physics), provide SEPARATE "box" for EACH option.
        3. NEVER combine multiple option images into one diagram_box.
        
        OUTPUT FORMAT:
        {
            "answer_key_table": null or {"1": "A", "2": "B", ...},
            "questions": [
                {
                    "q_id": "45",
                    "q_text": "Full question text without number",
                    "question_type": "mcq" or "numerical",
                    "diagram_box": [ymin, xmin, ymax, xmax] or null,
                    "options": [
                        "text option" OR {"text": "(A)", "box": [y1,x1,y2,x2]}
                    ]
                }
            ]
        }
        
        EXAMPLE - MCQ with image options:
        "options": [
            {"text": "(A)", "box": [200, 50, 350, 250]},
            {"text": "(B)", "box": [200, 260, 350, 460]},
            {"text": "(C)", "box": [360, 50, 510, 250]},
            {"text": "(D)", "box": [360, 260, 510, 460]}
        ]
        
        EXAMPLE - Numerical question:
        {"q_id": "26", "q_text": "Find the value of x...", "question_type": "numerical", "diagram_box": null, "options": []}
        """
        for attempt in range(MAX_RETRIES):
            try:
                response = model.generate_content([prompt, pil_image], generation_config={"response_mime_type": "application/json"})
                return json.loads(response.text)
            except Exception as e:
                time.sleep(1)
        return {"questions": []}

    def process_page(self, page_num):
        print(f"--- Page {page_num + 1} ---")
        page = self.doc[page_num]
        
        # High-Res Image for Vision
        pix = page.get_pixmap(dpi=RENDER_DPI)
        img_arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        cv2_img = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
        pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        data = self.get_gemini_analysis(pil_img)
        
        # FIX: Validate answer_key_table is a dict before updating
        answer_key = data.get("answer_key_table")
        if answer_key and isinstance(answer_key, dict):
            self.answer_key_map.update(answer_key)
        elif answer_key:
            print(f"  [WARN] Invalid answer_key_table format: {type(answer_key).__name__}")

        if not data.get("questions"): return

        for q in data["questions"]:
            try:
                # Robust parsing of Question ID
                q_num = int(re.search(r'\d+', str(q.get("q_id", "0"))).group())
            except: continue

            final_q = {
                "set": self.current_set,
                "grade": METADATA['grade'],
                "question_no": q_num,
                "question_type": q.get("question_type", "mcq"),  # mcq or numerical
                "question_txt": self.clean_q_text(q.get("q_text", "")),
                "options": [],
                "answer": None,
                "reference": self.current_ref,
                "image_question": "",
                "image_option": [],
                "validity": "Valid",
                "prepmode": METADATA['prepmode']
            }

            # ---------------------------
            # 1. QUESTION IMAGE HANDLING
            # ---------------------------
            if q.get("diagram_box"):
                # Naming: NEET_2022_Sample_1_7.png
                fname = f"{self.current_set}_{q_num}.png"
                save_path = os.path.join(self.set_dir, fname)
                
                # Priority 1: Raw Extract
                success = extract_raw_image_if_exists(self.doc, page, q["diagram_box"], save_path)
                # Priority 2: Smart Crop
                # Priority 2: Smart Crop
                if not success:
                    process_and_save_crop(cv2_img, q["diagram_box"], save_path)
                
                # --- REAL-TIME VALIDATION ---
                if self.validate_image_content(save_path):
                    final_q["image_question"] = fname
                else:
                    print(f"  [QUARANTINE] Logo/Junk detected: {fname}")
                    try: shutil.move(save_path, os.path.join(self.quarantine_dir, fname))
                    except: pass
                    final_q["image_question"] = None

            # ---------------------------
            # 2. OPTIONS HANDLING
            # ---------------------------
            raw_opts = q.get("options", [])
            
            # Global Check: Does ANY option in this question require an image?
            # If yes, we treat ALL options as image_option for consistency
            has_imgs = False
            for opt in raw_opts:
                if isinstance(opt, dict) and opt.get("box"):
                    has_imgs = True
                    break

            for i, opt in enumerate(raw_opts):
                if not opt: continue
                
                # Naming: NEET_2022_Sample_1_7_OPTION1.png
                fname = f"{self.current_set}_{q_num}_OPTION{i+1}.png"
                
                # Normalize Data (Fix for String vs Dict crash)
                if isinstance(opt, str):
                    opt_text = opt
                    opt_box = None
                else:
                    opt_text = opt.get("text", "")
                    opt_box = opt.get("box")

                if has_imgs:
                    # MIXED MODE: Populate image_option[]
                    if opt_box:
                        save_path = os.path.join(self.set_dir, fname)
                        if not extract_raw_image_if_exists(self.doc, page, opt_box, save_path):
                            process_and_save_crop(cv2_img, opt_box, save_path)
                        
                        # --- REAL-TIME VALIDATION ---
                        if self.validate_image_content(save_path):
                            final_q["image_option"].append(fname)
                        else:
                            print(f"  [QUARANTINE] Logo/Junk detected: {fname}")
                            try: shutil.move(save_path, os.path.join(self.quarantine_dir, fname))
                            except: pass
                            # Fallback to text if available
                            final_q["image_option"].append(self.clean_q_text(opt_text) if opt_text else "")

                    else:
                        # Text option in a mixed question
                        final_q["image_option"].append(self.clean_q_text(opt_text))
                else:
                    # TEXT ONLY MODE: Populate options[]
                    final_q["options"].append(self.clean_q_text(opt_text))

            self.final_json.append(final_q)
            print(f"  Saved Q{q_num}")

    def save(self):
        self.final_json.sort(key=lambda x: x["question_no"])
        
        if self.answer_key_map:
            for q in self.final_json:
                q["answer"] = self.answer_key_map.get(str(q["question_no"]))

        with open(os.path.join(self.set_dir, f"{self.current_set}.json"), 'w', encoding='utf-8') as f:
            json.dump(self.final_json, f, indent=4)
        print(f"\n[SUCCESS] Saved to {self.set_dir}")

    def validate_image_content(self, img_path):
        """
        Uses Gemini to check if the image is a valid diagram or just a logo.
        Returns True (keep) or False (discard).
        """
        try:
            if not os.path.exists(img_path): return False
            
            # Use a fast check
            check_prompt = """
            Analyze this image. Is it a valid educational diagram, graph, figure, or table for a Physics/Chemistry/Biology question?
            
            Reply ONLY "YES" or "NO".
            
            RULES:
            - If it is a Company Logo on its own, reply "NO".
            - If it is pure text clearly functioning as a header/footer, reply "NO".
            - If it is a circuit, chemical structure, graph, biological diagram, or table, reply "YES".
            - **CRITICAL:** If the image contains a valid diagram BUT has a background logo/watermark, reply "YES". The content is more important than the logo.
            """
            
            # Simple retry loop for validation
            for _ in range(2):
                try:
                    img_file = genai.upload_file(img_path)
                    while img_file.state.name == "PROCESSING":
                        time.sleep(1)
                        img_file = genai.get_file(img_file.name)
                        
                    response = model.generate_content([check_prompt, img_file])
                    text = response.text.strip().upper()
                    
                    if "YES" in text: return True
                    return False # Conservative: if not explicitly YES, kill it
                except:
                    time.sleep(1)
            return True # Fallback: keep on error
        except Exception as e:
            print(f"[WARNING] Validation failed for {img_path}: {e}")
            return True # Keep on error

    def run(self):
        for i in range(len(self.doc)):
        # for i in range(min(2, len(self.doc))):
            self.process_page(i)
            time.sleep(API_DELAY_SECONDS)
        
        # --- POST-PROCESSING VALIDATION STEP ---
        # (Validation now happens in real-time inside process_page)
        pass

        self.save()

def initialize_gemini():
    global model
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found.")
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-3-pro-image-preview')
    print("[INFO] Gemini API initialized.")

if __name__ == "__main__":
    initialize_gemini()
    PDF_FILE = "NEET_2025_PW.pdf"
    if os.path.exists(os.path.join(INPUT_FOLDER, PDF_FILE)):
        DirectExtractor(PDF_FILE).run()
    else:
        print("PDF not found.")
