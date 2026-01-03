import os
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
INPUT_FOLDER = r"raw_pdfs/JEE_PYQPs/2025"  # Adjusted path for cross-platform safety
OUTPUT_ROOT = "ROOT"

# METADATA: strictly follows SET_NAME_NUMBER convention
METADATA = {
    "set": "PCM_2025_SET_8",          # Example: Matches PHY_SET_1_7.png
    "grade": 12,
    "reference": "JEE_MAIN_2025_JAN_27_Shift2",
    "prepmode": "JEE",
    "subject": "PCM"
}

GARBAGE_PATTERNS = [
    r"Physics\s*-\s*Section\s*[A-Z]",
    r"Chemistry\s*-\s*Section\s*[A-Z]",
    r"Mathematics\s*-\s*Section\s*[A-Z]",
    r"MathonGo", r"JEE Main Previous Year Paper",
    r"Shift\s*\d", r"Page\s*:\s*\d+"
]

# Processing Constants
CROP_BUFFER = 35
WATERMARK_THRESHOLD = 210
RENDER_DPI = 300
API_DELAY_SECONDS = 2  # Increased slightly for stability
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
    # Convert to gray and threshold to find dark ink
    gray = cv2.cvtColor(loose_crop, cv2.COLOR_BGR2GRAY)
    _, ink_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Dilate to connect broken lines (like dashed graphs)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_mask = cv2.dilate(ink_mask, kernel, iterations=2)
    
    coords = cv2.findNonZero(dilated_mask)
    final_crop = loose_crop
    
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        if w > 10 and h > 10:
            # Snap to the exact ink bounds + tiny aesthetic pad
            pad = 5
            cy1 = max(0, y - pad)
            cx1 = max(0, x - pad)
            cy2 = min(loose_crop.shape[0], y + h + pad)
            cx2 = min(loose_crop.shape[1], x + w + pad)
            final_crop = loose_crop[cy1:cy2, cx1:cx2]

    # 3. CLEAN BACKGROUND (Watermark Removal)
    gray_final = cv2.cvtColor(final_crop, cv2.COLOR_BGR2GRAY)
    _, clean_mask = cv2.threshold(gray_final, WATERMARK_THRESHOLD, 255, cv2.THRESH_BINARY)
    # Force light pixels (watermarks) to pure white
    final_clean_img = np.where(clean_mask[..., None] == 255, 255, final_crop)

    cv2.imwrite(save_path, final_clean_img)
    return True

# ==========================================
# 3. MAIN EXTRACTION
# ==========================================
model = None 

class DirectExtractor:
    def __init__(self, pdf_filename):
        self.pdf_path = os.path.join(INPUT_FOLDER, pdf_filename)
        self.doc = fitz.open(self.pdf_path)
        
        # Strict Folder Structure
        self.set_dir = os.path.join(OUTPUT_ROOT, METADATA['prepmode'], METADATA['subject'], METADATA['set'])
        os.makedirs(self.set_dir, exist_ok=True)
        
        self.final_json = []
        self.answer_key_map = {}
        print(f"[OUTPUT] {self.set_dir}")

    def clean_q_text(self, text):
        if not text: return ""
        clean = " ".join(text.strip().split())
        for p in GARBAGE_PATTERNS: clean = re.sub(p, "", clean, flags=re.IGNORECASE)
        return re.sub(r'^(Q\s*)?\d+[\.\)\s]+', '', clean.strip()).strip()

    def get_gemini_analysis(self, pil_image):
        prompt = """
        Analyze this exam page and extract ALL questions. Return ONLY valid JSON.
        
        QUESTION NUMBER FORMATS (all valid):
        - "Q1", "q1", "Q.1", "1.", "1)", "(1)", "Question 1"
        - Extract just the NUMBER as "q_id" (e.g., "1", "45", "63")
        
        QUESTION TYPES:
        1. MCQ (Multiple Choice): Has 4 options (A/B/C/D or 1/2/3/4)
        2. NUMERICAL/FILL-IN-BLANK: No options, answer is a number. Set "options": [] and "question_type": "numerical"
        
        OPTION FORMATS (all valid):
        - "(A)", "(B)", "(C)", "(D)" 
        - "A.", "B.", "C.", "D."
        - "(1)", "(2)", "(3)", "(4)"
        - "1.", "2.", "3.", "4."
        - "a.", "b.", "c.", "d."
        
        CRITICAL IMAGE RULES:
        1. "diagram_box" is ONLY for figures/diagrams in the QUESTION TEXT itself.
        2. If OPTIONS contain graphs/figures, provide SEPARATE "box" for EACH option.
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
                "set": METADATA['set'],
                "grade": METADATA['grade'],
                "question_no": q_num,
                "question_type": q.get("question_type", "mcq"),  # mcq or numerical
                "question_txt": self.clean_q_text(q.get("q_text", "")),
                "options": [],
                "answer": None,
                "reference": METADATA['reference'],
                "image_question": "",
                "image_option": [],
                "validity": "Valid",
                "prepmode": METADATA['prepmode']
            }

            # ---------------------------
            # 1. QUESTION IMAGE HANDLING
            # ---------------------------
            if q.get("diagram_box"):
                # Naming: PCM_2025_SET_1_7.png
                fname = f"{METADATA['set']}_{q_num}.png"
                save_path = os.path.join(self.set_dir, fname)
                
                # Priority 1: Raw Extract
                success = extract_raw_image_if_exists(self.doc, page, q["diagram_box"], save_path)
                # Priority 2: Smart Crop
                if not success:
                    process_and_save_crop(cv2_img, q["diagram_box"], save_path)
                
                final_q["image_question"] = fname

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
                
                # Naming: PCM_2025_SET_1_7_OPTION1.png
                fname = f"{METADATA['set']}_{q_num}_OPTION{i+1}.png"
                
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
                        final_q["image_option"].append(fname)
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

        with open(os.path.join(self.set_dir, f"{METADATA['set']}.json"), 'w', encoding='utf-8') as f:
            json.dump(self.final_json, f, indent=4)
        print(f"\n[SUCCESS] Saved to {self.set_dir}")

    def run(self):
        for i in range(len(self.doc)):
            self.process_page(i)
            time.sleep(API_DELAY_SECONDS)
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
    PDF_FILE = "JEE_Main_2025_28_Jan_Shift_2.pdf"
    if os.path.exists(os.path.join(INPUT_FOLDER, PDF_FILE)):
        DirectExtractor(PDF_FILE).run()
    else:
        print("PDF not found.")