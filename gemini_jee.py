import os
import json
import re
import time
import cv2
import numpy as np
import fitz  # PyMuPDF
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

# ==========================================
# 1. CONFIGURATION
# ==========================================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Input/Output
INPUT_FOLDER = "raw_pdfs"
OUTPUT_ROOT = "ROOT"

METADATA = {
    "set": "PHY_SET_1",
    "grade": 12,
    "reference": "JEE_MAIN_2022_JULY_27",
    "prepmode": "JEE",
    "subject": "Physics"
}

# Garbage Filters (Removes headers, footers, and metadata from text)
GARBAGE_PATTERNS = [
    r"Physics\s*-\s*Section\s*[A-Z]",
    r"Chemistry\s*-\s*Section\s*[A-Z]",
    r"Mathematics\s*-\s*Section\s*[A-Z]",
    r"MathonGo", 
    r"JEE Main Previous Year Paper",
    r"JEE Main 2022",
    r"Contact Number:",
    r"Page\s*:\s*\d+",
    r"&neet prep",
    r"Question Paper",
    r"Shift\s*\d"
]


# ==========================================
# 2. VISION ENGINE: GAMMA WASH & SNAP
# ==========================================
def process_and_save_image(original_img, box_norm, save_path):
    """
    1. Gamma Correction: Washes out faint watermarks.
    2. Global Ink Snap: Crops to the exact outer limits of the black pixels.
    """
    if not box_norm or len(box_norm) != 4:
        return False

    h_img, w_img, _ = original_img.shape
    ymin, xmin, ymax, xmax = box_norm

    # 1. LOOSE CROP (Capture a wide buffer)
    # We grab MORE area to ensure we don't cut off labels like "Fig 1"
    buffer = 50 
    y1 = max(0, int((ymin / 1000) * h_img) - buffer)
    x1 = max(0, int((xmin / 1000) * w_img) - buffer)
    y2 = min(h_img, int((ymax / 1000) * h_img) + buffer)
    x2 = min(w_img, int((xmax / 1000) * w_img) + buffer)

    loose_crop = original_img[y1:y2, x1:x2]
    if loose_crop.size == 0: return False

    # 2. GAMMA WASH (The Magic Step)
    # This lightens mid-tones (watermarks) to white, keeps shadows (ink) dark.
    gamma = 2.0  # Higher = stronger washout of grey
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    washed = cv2.LUT(loose_crop, table)

    # 3. BINARY THRESHOLD
    # Convert to gray
    gray = cv2.cvtColor(washed, cv2.COLOR_BGR2GRAY)
    # Strict threshold: Only very dark things (ink) stay.
    # Any pixel > 180 brightness becomes White.
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # 4. GLOBAL BOUNDING BOX
    # Instead of finding specific shapes (contours), we find ALL ink pixels.
    coords = cv2.findNonZero(binary)
    
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        
        # If the found ink is too tiny (just noise/dust), ignore it
        if w > 10 and h > 10:
            # Add aesthetic padding
            pad = 8
            cy1 = max(0, y - pad)
            cx1 = max(0, x - pad)
            cy2 = min(loose_crop.shape[0], y + h + pad)
            cx2 = min(loose_crop.shape[1], x + w + pad)
            
            final_crop = loose_crop[cy1:cy2, cx1:cx2]
            
            # Save the original quality crop (not the washed one)
            cv2.imwrite(save_path, final_crop)
            return True

    # Fallback: If no ink found (rare), save the loose crop
    cv2.imwrite(save_path, loose_crop)
    return True

# # ==========================================
# # 2. VISION ENGINE: CROP & CLEAN
# # ==========================================
# def process_and_save_image(original_img, box_norm, save_path):
#     """
#     1. Removes Watermarks (forces light grey -> white).
#     2. Finds exact ink boundaries.
#     3. Crops tightly around the ink.
#     """
#     if not box_norm or len(box_norm) != 4:
#         return False

#     h_img, w_img, _ = original_img.shape
#     ymin, xmin, ymax, xmax = box_norm

#     # 1. LOOSE CROP (Capture a safe buffer area first)
#     buffer = 35 
#     y1 = max(0, int((ymin / 1000) * h_img) - buffer)
#     x1 = max(0, int((xmin / 1000) * w_img) - buffer)
#     y2 = min(h_img, int((ymax / 1000) * h_img) + buffer)
#     x2 = min(w_img, int((xmax / 1000) * w_img) + buffer)

#     loose_crop = original_img[y1:y2, x1:x2]
#     if loose_crop.size == 0: return False

#     # 2. DIGITAL CLEANING (Remove Watermarks)
#     # Convert to grayscale to check brightness
#     gray = cv2.cvtColor(loose_crop, cv2.COLOR_BGR2GRAY)
    
#     # Threshold: Anything lighter than 200 (watermarks are usually ~210-230) becomes WHITE
#     # Anything darker (ink is ~0-50) stays dark.
#     # This creates a mask for the ink.
#     _, ink_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

#     # 3. FIND BOUNDING BOX OF INK (Tight Crop)
#     # Dilate slightly to connect broken lines (like dashed graphs)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     dilated_mask = cv2.dilate(ink_mask, kernel, iterations=2)
    
#     contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     final_crop = loose_crop # Default to loose crop if no contours found
    
#     if contours:
#         # Find the super-box covering all ink contours
#         min_x, min_y = float('inf'), float('inf')
#         max_x, max_y = 0, 0
#         found_ink = False

#         for cnt in contours:
#             x, y, w, h = cv2.boundingRect(cnt)
#             if w * h > 30: # Filter dust/noise
#                 found_ink = True
#                 min_x = min(min_x, x)
#                 min_y = min(min_y, y)
#                 max_x = max(max_x, x + w)
#                 max_y = max(max_y, y + h)

#         if found_ink:
#             # Add aesthetic padding
#             pad = 5
#             cy1 = max(0, min_y - pad)
#             cx1 = max(0, min_x - pad)
#             cy2 = min(loose_crop.shape[0], max_y + pad)
#             cx2 = min(loose_crop.shape[1], max_x + pad)
#             final_crop = loose_crop[cy1:cy2, cx1:cx2]

#     # 4. FINAL CLEAN & SAVE
#     # We want to save a clean version where the background is pure white
#     # Use the mask to force non-ink pixels to white
    
#     # Re-calculate mask for the FINAL tight crop
#     gray_final = cv2.cvtColor(final_crop, cv2.COLOR_BGR2GRAY)
#     _, clean_mask = cv2.threshold(gray_final, 200, 255, cv2.THRESH_BINARY) 
#     # clean_mask: Background is White (255), Ink is Black (0) due to threshold logic
#     # Actually: threshold(200, 255, BINARY) -> >200 is 255 (White), <200 is 0 (Black)
    
#     # Create white background image
#     white_bg = np.ones_like(final_crop) * 255
    
#     # Copy only the dark pixels (ink) from original to white background
#     # This effectively erases the light grey watermark
#     # Usage: where mask is black (ink), use image; else use white
#     final_clean_img = cv2.bitwise_or(final_crop, final_crop, mask=cv2.bitwise_not(clean_mask))
#     # Invert logic: Make background white
#     final_clean_img = np.where(clean_mask[..., None] == 255, 255, final_crop)

#     cv2.imwrite(save_path, final_clean_img)
#     return True

# ==========================================
# 3. GEMINI EXTRACTION LOGIC
# ==========================================
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found.")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-3-pro-image-preview') 

class DirectExtractor:
    def __init__(self, pdf_filename):
        self.pdf_path = os.path.join(INPUT_FOLDER, pdf_filename)
        self.doc = fitz.open(self.pdf_path)
        self.set_dir = os.path.join(OUTPUT_ROOT, METADATA['prepmode'], METADATA['subject'], METADATA['set'])
        os.makedirs(self.set_dir, exist_ok=True)
        self.final_json = []
        self.answer_key_map = {}

    def clean_q_text(self, text):
        """Strips Question Numbers and Garbage."""
        if not text: return ""
        clean = " ".join(text.strip().split())
        for pattern in GARBAGE_PATTERNS:
            clean = re.sub(pattern, "", clean, flags=re.IGNORECASE)
        
        # Remove "Q1.", "1.", "(1)" from start
        clean = re.sub(r'^(Q\s*)?\d+[\.\)\s]+', '', clean.strip())
        return clean.strip()

    def get_gemini_analysis(self, pil_image, page_num):
        prompt = """
        Analyze this exam page. Return valid JSON.
        
        TASK 1: Extract Questions
        - "q_id": Actual number printed (e.g. "63").
        - "q_text": Full question text. Include statements/lists (a, b...) here.
        - "diagram_box": [ymin, xmin, ymax, xmax] (0-1000) for diagrams/tables.
        - "options": List of 4 options.
           - If text: {"text": "Option content", "box": null}
           - If image/mixed: {"text": null, "box": [ymin, xmin, ymax, xmax]}
        
        TASK 2: Answer Key
        - If "Answer Key" table exists, extract it into "answer_key_table": {"63": "A", ...} and set "questions": [].
        
        OUTPUT JSON:
        {"answer_key_table": null, "questions": []}
        """
        try:
            response = model.generate_content([prompt, pil_image], generation_config={"response_mime_type": "application/json"})
            return json.loads(response.text)
        except:
            return {"questions": []}

    def process_page(self, page_num):
        print(f"--- Page {page_num + 1} ---")
        page = self.doc[page_num]
        pix = page.get_pixmap(dpi=300)
        img_arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        cv2_img = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
        pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        data = self.get_gemini_analysis(pil_img, page_num)
        
        if data.get("answer_key_table"):
            print("  [INFO] Answer Key Found!")
            self.answer_key_map.update(data["answer_key_table"])
            return

        if not data.get("questions"): return

        for q in data["questions"]:
            raw_id = str(q.get("q_id", "0"))
            try:
                q_num_int = int(re.search(r'\d+', raw_id).group())
            except:
                q_num_int = 0

            final_q = {
                "set": METADATA['set'],
                "grade": METADATA['grade'],
                "question_no": q_num_int,
                "question_txt": self.clean_q_text(q.get("q_text", "")),
                "options": [],
                "image_option": [],
                "answer": None,
                "image_question": "",
                "reference": METADATA['reference'],
                "validity": "Valid",
                "prepmode": METADATA['prepmode']
            }

            # Question Image
            if q.get("diagram_box"):
                fname = f"{METADATA['set']}_{raw_id}.png"
                if process_and_save_image(cv2_img, q["diagram_box"], os.path.join(self.set_dir, fname)):
                    final_q["image_question"] = fname

            # Options
            raw_opts = q.get("options", [])
            is_complex = any(opt.get("box") for opt in raw_opts if opt)
            
            for i, opt in enumerate(raw_opts):
                if not opt: continue
                if is_complex:
                    if opt.get("box"):
                        fname = f"{METADATA['set']}_{raw_id}_OPT_{i+1}.png"
                        process_and_save_image(cv2_img, opt["box"], os.path.join(self.set_dir, fname))
                        final_q["image_option"].append(fname)
                    else:
                        final_q["image_option"].append(self.clean_q_text(opt.get("text", "")))
                else:
                    final_q["options"].append(self.clean_q_text(opt.get("text", "")))

            self.final_json.append(final_q)
            print(f"  Saved Q{raw_id} | Img: {final_q['image_question']}")

    def finalize_answers(self):
        if self.answer_key_map:
            print(f"\n[INFO] Mapping {len(self.answer_key_map)} answers...")
            for q in self.final_json:
                qid = str(q["question_no"])
                if qid in self.answer_key_map:
                    q["answer"] = self.answer_key_map[qid]

    def save(self):
        self.final_json.sort(key=lambda x: x["question_no"])
        with open(os.path.join(self.set_dir, f"{METADATA['set']}.json"), 'w', encoding='utf-8') as f:
            json.dump(self.final_json, f, indent=4)
        print("\n[SUCCESS] Saved.")

    def run(self):
        for i in range(len(self.doc)):
            self.process_page(i)
            time.sleep(1)
        self.finalize_answers()
        self.save()

if __name__ == "__main__":
    PDF_FILE = "JEE_Main_2022_27_Jul_Shift_1.pdf" 
    if os.path.exists(os.path.join(INPUT_FOLDER, PDF_FILE)):
        DirectExtractor(PDF_FILE).run()
    else:
        print("PDF not found.")