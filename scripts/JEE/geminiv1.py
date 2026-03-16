import os
import json
import re
import time
import shutil          
import cv2
import numpy as np
import fitz             # PyMuPDF
import io
import sys
from google import genai
from google.genai import types
from PIL import Image
from dotenv import load_dotenv

# ==========================================
# 1. CONFIGURATION
# ==========================================

# Resolve project root (2 levels up from scripts/JEE/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Input/Output (resolved from project root)
INPUT_FOLDER = os.path.join(PROJECT_ROOT, "raw_pdfs", "JEE_PYQPs")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "ROOT")

# METADATA: fallback only — overridden per-config in __main__
METADATA = {
    "set": "PCM_2025_SET_1",
    "grade": 12,
    "reference": "JEE_MAIN_2025_JAN_22_Shift1",
    "prepmode": "JEE",
    "subject": "PCM"
}

GARBAGE_PATTERNS = [
    r"Physics\s*-\s*Section\s*[A-Z]",
    r"Chemistry\s*-\s*Section\s*[A-Z]",
    r"Mathematics\s*-\s*Section\s*[A-Z]",
    r"MathonGo", r"JEE Main Previous Year Paper",
    r"Shift\s*\d", r"Page\s*:\s*\d+",
    r"JEE Main 2025 January", r"2025\s*\(\d+\s*Jan\s*Shift\s*\d+\)"
]

# Processing Constants
CROP_BUFFER = 70
WATERMARK_THRESHOLD = 210
RENDER_DPI = 300
API_DELAY_SECONDS = 2
MAX_RETRIES = 3

MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
}

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
                if intersect.is_empty:
                    continue

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
    PRIORITY 2: Padded crop with watermark removal.
    """
    if box_norm is None or len(box_norm) != 4:
        return False

    h_img, w_img, _ = original_img.shape
    ymin, xmin, ymax, xmax = box_norm

    buffer = 15

    y1 = max(0, int((ymin / 1000) * h_img) - buffer)
    x1 = max(0, int((xmin / 1000) * w_img) - buffer)
    y2 = min(h_img, int((ymax / 1000) * h_img) + buffer)
    x2 = min(w_img, int((xmax / 1000) * w_img) + buffer)

    final_crop = original_img[y1:y2, x1:x2]
    if final_crop.size == 0:
        return False

    # Watermark removal
    gray_final = cv2.cvtColor(final_crop, cv2.COLOR_BGR2GRAY)
    _, clean_mask = cv2.threshold(gray_final, WATERMARK_THRESHOLD, 255, cv2.THRESH_BINARY)
    final_clean_img = np.where(clean_mask[..., None] == 255, 255, final_crop)

    cv2.imwrite(save_path, final_clean_img)
    return True

def stitch_images(existing_path, new_img_arr, save_path):
    """Vertically stacks existing image and new image array."""
    try:
        if not os.path.exists(existing_path):
            return False

        img1 = cv2.imread(existing_path)
        img2 = new_img_arr

        h1, w1, _ = img1.shape
        h2, w2, _ = img2.shape
        max_w = max(w1, w2)

        canvas = np.full((h1 + h2 + 10, max_w, 3), 255, dtype=np.uint8)
        canvas[0:h1, 0:w1] = img1
        canvas[h1 + 10:h1 + 10 + h2, 0:w2] = img2

        cv2.imwrite(save_path, canvas)
        return True
    except Exception as e:
        print(f"[WARN] Stitching failed: {e}")
        return False

# ==========================================
# NUMERICAL QUESTION CONSTRAINTS BY YEAR
# ==========================================
def get_numerical_limit_for_year(year):
    """
    Returns the maximum number of numerical questions allowed per subject for a given year.
    
    - 2012-2019: 0 numerical questions (90 total questions)
    - 2020: 5 numerical questions per subject (90 total questions)
    - 2021-2024: 10 numerical questions per subject (90 total questions)
    - 2025: 5 numerical questions per subject (75 total questions - REDUCED)
    """
    if year < 2020:
        return 0
    elif year == 2025:
        return 5
    elif year == 2020:
        return 5
    else:  # 2021-2024
        return 10

# ==========================================
# 3. MAIN EXTRACTION
# ==========================================
client = None

class DirectExtractor:
    def __init__(self, pdf_file, metadata=None):
        self.pdf_file = pdf_file
        self.metadata = metadata if metadata else METADATA.copy()

        # Extract year from metadata
        self.year = self.metadata.get("year", 2025)
        
        # Set numerical question limit based on year
        self.numerical_limit_per_subject = get_numerical_limit_for_year(self.year)
        
        # Track numerical questions per subject
        self.numerical_count = {
            "PHY": 0,
            "CHEM": 0,
            "MATH": 0
        }
        
        print(f"[INFO] Year: {self.year}, Numerical limit per subject: {self.numerical_limit_per_subject}")

        self.current_set = self.metadata.get("set", METADATA["set"])
        self.current_ref = self.metadata.get("reference", METADATA["reference"])

        self.set_dir = os.path.join(OUTPUT_ROOT, self.metadata['prepmode'],
                                    self.metadata['subject'], self.current_set)
        os.makedirs(self.set_dir, exist_ok=True)

        if os.path.exists(pdf_file):
            pdf_path = pdf_file                                          # direct full path
        elif os.path.exists(os.path.join(INPUT_FOLDER, pdf_file)):
            pdf_path = os.path.join(INPUT_FOLDER, pdf_file)             # relative to INPUT_FOLDER
        else:
            raise FileNotFoundError(f"PDF not found: {pdf_file}")

        self.doc = fitz.open(pdf_path)
        self.final_json = []
        self.answer_key_map = {}
        print(f"[OUTPUT] {self.set_dir}")

    # ----------------------------
    # Text-cleaning helpers
    # ----------------------------
    def clean_q_text(self, text):
        if not text:
            return ""
        if "dummy text" in text.lower() or "hello dummy" in text.lower():
            return ""
        clean = " ".join(text.strip().split())
        for p in GARBAGE_PATTERNS:
            clean = re.sub(p, "", clean, flags=re.IGNORECASE)
        return re.sub(r'^(Q\s*)?\d+[\.\)]\s*', '', clean.strip()).strip()

    def clean_option_text(self, text):
        """Strip ONLY numeric option labels: (1), 1., 2), [3], etc.

        Single-letter tokens like (A), (B), A) are deliberately
        left alone — in JEE Assertion/Reason questions they are
        content references, not labels we added.
        """
        if not text:
            return ""
        return re.sub(r'^\s*[\(\[]?\s*\d+\s*[\)\]\.]\s*', '', text).strip()

    def clean_json(self, text):
        """Fixes common JSON errors from LLMs, especially LaTeX backslashes."""
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```', '', text)
        text = text.strip()
        # Escape backslashes NOT followed by valid JSON escape chars
        text = re.sub(r'\\(?![\\/u"bfnrt])', r'\\\\', text)
        # Escape \r, \n, \t, \b, \f when followed by more letters (LaTeX commands)
        text = re.sub(r'\\([rnbtf])(?=[a-zA-Z])', r'\\\\\1', text)
        return text

    # ----------------------------------------------------------
    # Gemini call
    # ----------------------------------------------------------
    def get_gemini_analysis(self, pil_image):
        prompt = r"""
        Analyze this exam page and extract ALL questions. Return ONLY valid JSON.

        CRITICAL: IGNORE HEADERS AND FOOTERS.
        - Ignore text like "MathonGo", "JEE Main 2025", Page numbers, or Shift details.
        - Do NOT treat them as part of the question text.

        MATH FORMATTING (CRITICAL - STRICT LATEX ONLY):
        - NO UNICODE MATH SYMBOLS (like ×, ÷, ≈, ≤, ≥). USE ONLY LATEX.
        - Wrap inline math in single $ delimiters: $x^2 + y^2 = r^2$
        - Use LaTeX commands: $\frac{a}{b}$, $\sqrt{x}$, $\int_0^1$, $\sum_{n=1}^{\infty}$
        - Subscripts: $a_1$, $x_n$ NOT a₁, xₙ
        - Superscripts: $x^2$, $e^x$ NOT x², eˣ
        - Greek letters: $\alpha$, $\beta$, $\pi$ NOT α, β, π
        - Chemical formulas: $H_2O$

        SUBJECT CLASSIFICATION:
        - Identify the subject for EACH question: "Physics", "Chemistry", or "Mathematics".
        - Return this in the "subject" field.

        TABLES & DIAGRAMS:
        - COMPLEX TABLES (like Match List I & II): Mark them as a "diagram_box" so they become an IMAGE.
        - This is mandatory for "Match List" questions.
        
        MATCH-LIST QUESTIONS (CRITICAL):
        - These have a TWO-COLUMN TABLE (List-I and List-II)
        - BELOW the table are numbered options like: (1) A-III, B-IV, C-I, D-II
        - The TABLE and OPTIONS may be on DIFFERENT PAGES
        - You MUST extract these numbered options as text in the "options" array
        - Example: ["(1) A-III, B-IV, C-I, D-II", "(2) A-IV, B-II, C-I, D-III", ...]
        - Also set diagram_box to capture the table as an image

        MULTI-PART MCQ (Select Correct Statements):
        - If question lists lettered sub-statements: A. ..., B. ..., C. ..., D. ...
        - Followed by numbered options: (1) A and B only, (2) B, C and D only...
        - Extract ONLY the numbered options (1), (2), (3), (4) in "options" array
        - Include ALL sub-statements in "q_text"

        FIGURE/DIAGRAM DETECTION:
        - If question text mentions "figure", "diagram", "graph", "as shown", 
          "shown in figure", "in the figure", ALWAYS set diagram_box to capture the visual.
        - Even if the diagram is not explicitly labeled, capture it.

        CONTINUATION FROM PREVIOUS PAGE (VERY CRITICAL):
        - If the page starts with content belonging to a previous question (no new Q number), set "q_id": "CONT"
        - This includes:
          * Additional text paragraphs
          * Diagram/table that continues from previous page
          * OPTIONS for a question (especially match-list questions)
        - MATCH-LIST OPTIONS DETECTION:
          * If you see numbered options like (1) A-III, B-IV, C-I, D-II
          * But NO question text or Q-number above them
          * These are OPTIONS from a match-list question on the previous page
          * Set "q_id": "CONT" and return them in "options" array
          * Example: {"q_id": "CONT", "options": ["(1) A-III, B-IV...", "(2) A-IV, B-II...", ...]}

        QUESTION TYPES (UPPERCASE ONLY):
        - "MCQ": Multiple Choice (Single Correct)
        - "MSQ": Multiple Select (One or More Correct)
        - "NUMERICAL": Integer/Decimal answer (No options)

        OPTION TEXT / IMAGES:
        - If option is text/math: Return string.
        - If option is an IMAGE/GRAPH/CHEMICAL STRUCTURE:
          Return Object {"text": "", "box": [ymin, xmin, ymax, xmax]}.
        - CRITICAL: "box" MUST cover the ENTIRE option (Text Label + Diagram).
        - If options reference diagrams ABOVE them (e.g. "B and C only" referring 
          to diagrams labeled B, C in the question), those diagrams belong in 
          diagram_box (question image), NOT in option boxes.

        ANSWER KEY TABLE:
        - Look for a table titled "ANSWER KEYS" with question numbers and answers.
        - Extract as "answer_key_table": {"1": "4", "2": "3", ...}
        - If NO answer key table on this page, return null.

        IMPORTANT: Even if a page contains ONLY an answer-key table and no new
        questions, still return it inside the standard wrapper so it is not lost:
          { "questions": [], "answer_key_table": { ... } }

        OUTPUT FORMAT:
        {
            "questions": [
                {
                    "q_id": "45",
                    "subject": "Physics",
                    "q_text": "Calculate force...",
                    "question_type": "MCQ",
                    "diagram_box": [ymin, xmin, ymax, xmax] or null,
                    "options": ["text option", ...]
                }
            ],
            "answer_key_table": {"1": "4", "2": "3", ...} or null
        }
        """
        for attempt in range(MAX_RETRIES):
            try:
                response = client.models.generate_content(
                    model='gemini-3-flash-preview',  # Stable model
                    contents=[prompt, pil_image],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
                raw_text = response.text
                print(f"  [DEBUG] Raw Gemini response (len={len(raw_text)}): {raw_text[:200]}...")
                parsed = None

                # Escalating JSON repair
                try:
                    parsed = json.loads(raw_text)
                except json.JSONDecodeError:
                    cleaned = self.clean_json(raw_text)
                    try:
                        parsed = json.loads(cleaned)
                    except json.JSONDecodeError:
                        fixed = cleaned.replace(r'\frac', r'\\frac') \
                                       .replace(r'\text', r'\\text') \
                                       .replace(r'\sqrt', r'\\sqrt') \
                                       .replace(r'\int', r'\\int') \
                                       .replace(r'\sum', r'\\sum') \
                                       .replace(r'\alpha', r'\\alpha') \
                                       .replace(r'\beta', r'\\beta') \
                                       .replace(r'\gamma', r'\\gamma') \
                                       .replace(r'\theta', r'\\theta') \
                                       .replace(r'\pi', r'\\pi') \
                                       .replace(r'\Delta', r'\\Delta') \
                                       .replace(r'\times', r'\\times') \
                                       .replace(r'\cdot', r'\\cdot') \
                                       .replace(r'\infty', r'\\infty')
                        try:
                            parsed = json.loads(fixed)
                        except json.JSONDecodeError:
                            blind_fix = cleaned.replace('\\', '\\\\')
                            parsed = json.loads(blind_fix)

                normalized_q = []
                answer_key_on_page = None

                if isinstance(parsed, list):
                    if len(parsed) > 0 and isinstance(parsed[0], list):
                        normalized_q = parsed[0]
                    else:
                        normalized_q = parsed
                elif isinstance(parsed, dict):
                    raw_questions = parsed.get("questions", [])
                    if isinstance(raw_questions, list) and len(raw_questions) > 0 \
                       and isinstance(raw_questions[0], list):
                        normalized_q = raw_questions[0]
                    else:
                        normalized_q = raw_questions
                    answer_key_on_page = parsed.get("answer_key_table")

                # Count genuinely populated questions
                valid_q_count = 0
                for q in normalized_q:
                    if isinstance(q, dict) and (q.get("q_text") or q.get("options") or q.get("diagram_box")):
                        valid_q_count += 1

                # Page is acceptable if it has questions OR an answer key
                has_answer_key = (answer_key_on_page and isinstance(answer_key_on_page, dict)
                                  and len(answer_key_on_page) > 0)

                if valid_q_count == 0 and not has_answer_key:
                    raise ValueError("Gemini returned only GHOST questions (Empty Content) and no answer key.")

                return parsed

            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "ResourceExhausted" in err_str or "quota" in err_str.lower():
                    print(f"\n[CRITICAL] API QUOTA EXCEEDED. Stopping script immediately.")
                    sys.exit(1)

                print(f"  [gemini_err] {e} (Retrying {attempt + 1}/{MAX_RETRIES})...")
                time.sleep(1 + attempt)

        # All retries exhausted — return empty but structurally valid
        return {"questions": [], "answer_key_table": None}

    # ----------------------------------------------------------
    # 4. NUMERICAL VALIDATION LOGIC
    # ----------------------------------------------------------
    def determine_subject(self, qid):
        """
        Determines the subject code based on QID and Year.
        Matches the logic used in save().
        """
        try:
            qid = int(qid)
        except:
            return "UNKNOWN"
            
        if self.year == 2020: # 75-question format (P-C-M)
            if 1 <= qid <= 25: return "PHY"
            elif 26 <= qid <= 50: return "CHEM"
            else: return "MATH"
        elif self.year >= 2025:  # 75-question format (M-P-C based on existing save logic)
            if 1 <= qid <= 25: return "MATH"
            elif 26 <= qid <= 50: return "PHY"
            else: return "CHEM"
        else:  # 90-question format (P-C-M)
            if 1 <= qid <= 30: return "PHY"
            elif 31 <= qid <= 60: return "CHEM"
            else: return "MATH"

    def get_expected_numerical_ranges(self):
        """
        Returns the expected question number ranges for numerical questions based on year.
        Adjusted to match the subject ordering in determine_subject.
        """
        if self.year == 2020:
             # 75 questions: 25 per subject
             # P-C-M order
             return {
                "PHY":  range(21, 26),   # Q21-25
                "CHEM": range(46, 51),   # Q46-50
                "MATH": range(71, 76)    # Q71-75
             }
        elif self.year >= 2025:
            # 75 questions: 25 per subject
            # Ordering appears to be MATH (1-25), PHY (26-50), CHEM (51-75) in save()
            if self.numerical_limit_per_subject == 5:
                return {
                    "MATH": range(21, 26),   # Q21-25
                    "PHY":  range(46, 51),   # Q46-50
                    "CHEM": range(71, 76)    # Q71-75
                }
            else:
                return {"PHY": range(0, 0), "CHEM": range(0, 0), "MATH": range(0, 0)}
        else:
            # 90 questions: 30 per subject (2012-2024, excluding 2020)
            # Ordering: PHY (1-30), CHEM (31-60), MATH (61-90)
            if self.numerical_limit_per_subject == 10:
                return {
                    "PHY": range(21, 31),   # Q21-30
                    "CHEM": range(51, 61),  # Q51-60
                    "MATH": range(81, 91)   # Q81-90
                }
            elif self.numerical_limit_per_subject == 5:
                # Fallback for other years if limit is set to 5 (e.g. maybe 2021 had optional)
                # But typically 2021-2024 is 10.
                return {
                    "PHY": range(26, 31),   # Q26-30
                    "CHEM": range(56, 61),  # Q56-60
                    "MATH": range(86, 91)   # Q86-90
                }
            else:  # 0 numericals
                return {"PHY": range(0, 0), "CHEM": range(0, 0), "MATH": range(0, 0)}

    def validate_numerical_question(self, question_data):
        """
        Validates if a numerical question should be accepted based on:
        1. Year-based count constraints
        2. Position within the section (numericals only at the end)
        
        Returns: (is_valid, warning_message)
        """
        qt = question_data.get("question_type", "").lower()
        
        # If NOT numerical, auto-pass
        if qt != "numerical":
            return True, None
        
        q_no = question_data.get("question_no", 0)
        subject_code = self.determine_subject(q_no)
        
        # Get expected numerical ranges
        numerical_ranges = self.get_expected_numerical_ranges()
        expected_range = numerical_ranges.get(subject_code, range(0, 0))
        
        # Check 1: Count limit
        if self.numerical_count.get(subject_code, 0) >= self.numerical_limit_per_subject:
            warning = (f"[WARNING] Numerical question limit exceeded for {subject_code} "
                      f"in year {self.year} (limit: {self.numerical_limit_per_subject}). "
                      f"Question #{q_no} may be incorrectly classified.")
            return False, warning
        
        # Check 2: Position validation
        if q_no not in expected_range:
            warning = (f"[WARNING] Numerical question Q{q_no} ({subject_code}) is outside "
                      f"expected range {list(expected_range)} for year {self.year}. "
                      f"Numericals should only appear at the END of each section.")
            return False, warning
        
        # Increment counter
        self.numerical_count[subject_code] += 1
        return True, None

    def add_question_to_json(self, question_data):
        """
        Adds a question to final_json with numerical question validation.
        """
        # Inject subject_code for proper validation context if not present
        if "subject_code" not in question_data:
             question_data["subject_code"] = self.determine_subject(question_data.get("question_no", 0))

        is_valid, warning = self.validate_numerical_question(question_data)
        
        if warning:
            print(warning)
            
            # Convert to MCQ if limit exceeded or position invalid
            if not is_valid:
                print(f"[FIX] Converting Q#{question_data.get('question_no')} from Numerical to MCQ")
                question_data["question_type"] = "mcq"
        
        self.final_json.append(question_data)

    # -------------------------
    # Single-page processor
    # -------------------------
    def process_page(self, page_num):
        print(f"--- Page {page_num + 1} ---")
        page = self.doc[page_num]

        pix = page.get_pixmap(dpi=RENDER_DPI)
        img_arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        cv2_img = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
        pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        data = self.get_gemini_analysis(pil_img)

        # Normalize
        questions_list = []

        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], list):
                questions_list = data[0]
            else:
                questions_list = data
        elif isinstance(data, dict):
            raw_questions = data.get("questions", [])
            if isinstance(raw_questions, list) and len(raw_questions) > 0 \
               and isinstance(raw_questions[0], list):
                questions_list = raw_questions[0]
            else:
                questions_list = raw_questions

        # # ---------------------------------------------------------------

            # Extract Answer Key (works even when questions_list is empty)
            answer_key = data.get("answer_key_table")
            if answer_key and isinstance(answer_key, dict):
                self.answer_key_map.update(answer_key)
                print(f"  [INFO] Extracted {len(answer_key)} answers from Answer Key table.")
            elif answer_key:
                print(f"  [WARN] Invalid answer_key_table format: {type(answer_key).__name__}")

        if not questions_list:
            print(f"[INFO] No questions on Page {page_num + 1} (may be answer-key only).")
            return

        # ----- per-question loop -----
        for q in questions_list:
            if not isinstance(q, dict):
                print(f"  [WARN] Skipping non-dict item: {type(q).__name__}")
                continue

            q_text = q.get('q_text') or ''
            q_opts = q.get('options') or []
            print(f"    [DEBUG] Raw QID: {q.get('q_id')} | Text len: {len(q_text)} | Options: {len(q_opts)}")

            qid_raw = str(q.get("q_id", "0"))

            # =====================================================
            # CONTINUATION handling
            # =====================================================
            if qid_raw == "CONT" or "CONT" in qid_raw.upper():
                if self.final_json:
                    try:
                        prev_q = self.final_json[-1]
                        print(f"  [MERGE] Merging content into Q{prev_q['question_no']}")

                        more_text = self.clean_q_text(q.get("q_text", ""))
                        if more_text:
                            prev_q["question_txt"] += " " + more_text

                        if q.get("options"):
                            raw_opts_cont = q.get("options") or []
                            current_opt_count = len(prev_q.get("options", []))

                            for i, opt in enumerate(raw_opts_cont):
                                opt_text = ""
                                opt_box = None
                                if isinstance(opt, str):
                                    opt_text = opt
                                elif isinstance(opt, dict):
                                    opt_text = opt.get("text", "")
                                    opt_box = opt.get("box")
                                else:
                                    continue

                                label_idx = current_opt_count + i
                                label_prefix = f"{chr(65 + label_idx)}) "
                                clean_text = self.clean_option_text(opt_text)
                                full_text = label_prefix + clean_text

                                prev_q["options"].append(full_text)

                                if opt_box:
                                    fname = f"{self.current_set}_{prev_q['question_no']}_OPTION{label_idx + 1}.png"
                                    save_path = os.path.join(self.set_dir, fname)

                                    if not extract_raw_image_if_exists(self.doc, page, opt_box, save_path):
                                        process_and_save_crop(cv2_img, opt_box, save_path)

                                    if os.path.exists(save_path):
                                        if "image_option" not in prev_q:
                                            prev_q["image_option"] = []
                                        prev_q["image_option"].append(fname)
                                        print(f"    [CONT] Saved option image {fname}")
                                    else:
                                        if "image_option" not in prev_q:
                                            prev_q["image_option"] = []
                                        prev_q["image_option"].append(full_text)
                                else:
                                    if prev_q.get("image_option"):
                                        prev_q["image_option"].append(full_text)

                        # Diagram stitching
                        if q.get("diagram_box"):
                            fname = f"{self.current_set}_{prev_q['question_no']}.png"
                            save_path = os.path.join(self.set_dir, fname)
                            temp_path = os.path.join(self.set_dir, "temp_stitch.png")
                            new_crop_success = False

                            if extract_raw_image_if_exists(self.doc, page, q["diagram_box"], temp_path):
                                new_crop_success = True
                            elif process_and_save_crop(cv2_img, q["diagram_box"], temp_path):
                                new_crop_success = True

                            if new_crop_success:
                                if prev_q["image_question"] and os.path.exists(save_path):
                                    temp_img = cv2.imread(temp_path)
                                    stitch_images(save_path, temp_img, save_path)
                                    print(f"    [STITCH] Merged diagram into {fname}")
                                else:
                                    if os.path.exists(save_path):
                                        os.remove(save_path)
                                    os.rename(temp_path, save_path)
                                    prev_q["image_question"] = fname
                                    print(f"    [CONT] Saved new diagram {fname}")

                            # FIX #8: always clean up temp file
                            if os.path.exists(temp_path):
                                os.remove(temp_path)

                    except Exception as e:
                        print(f"  [WARN] CONT Merge failed: {e}")
                    continue
                else:
                    print(f"  [WARN] 'CONT' on page {page_num + 1} but no previous Q. Skipping.")
                    continue

            # =====================
            # Parse question ID
            # =====================
            q_num = 0
            try:
                q_num = int(re.search(r'\d+', qid_raw).group())
            except Exception:
                pass

            if q_num == 0:
                if self.final_json:
                    last_num = self.final_json[-1]["question_no"]
                    q_num = last_num + 1
                    print(f"  [FIX] QID 0 → auto-assigned Q{q_num}")
                else:
                    print(f"  [WARN] Skipped invalid QID: {qid_raw}")
                    continue

            raw_type = q.get("question_type", "MCQ").upper()
            if raw_type == "NUMERICAL":
                final_type = "Numerical"
            elif raw_type == "MSQ":
                final_type = "msq"
            else:                          # MCQ or anything else
                final_type = "mcq"

            final_q = {
                "set":             self.current_set,
                "grade":           self.metadata['grade'],
                "question_no":     q_num,
                "question_type":   final_type,
                "question_txt":    self.clean_q_text(q.get("q_text", "")),
                "options":         [],
                "answer":          None,
                "reference":       self.current_ref,
                "subject":         q.get("subject", "Unknown"),
                "image_question":  "",
                "image_option":    [],
                "validity":        "Valid",
                "prepmode":        self.metadata['prepmode']
            }

            # --- Question image ---
            if q.get("diagram_box"):
                fname = f"{self.current_set}_{q_num}.png"
                save_path = os.path.join(self.set_dir, fname)

                if not extract_raw_image_if_exists(self.doc, page, q["diagram_box"], save_path):
                    process_and_save_crop(cv2_img, q["diagram_box"], save_path)

                if os.path.exists(save_path):
                    final_q["image_question"] = fname

            # --- Options ---
            raw_opts = q.get("options") or []

            # Does any option carry an image?
            has_imgs = any(isinstance(opt, dict) and opt.get("box") for opt in raw_opts)

            for i, opt in enumerate(raw_opts):
                if not opt:
                    continue

                fname = f"{self.current_set}_{q_num}_OPTION{i + 1}.png"

                if isinstance(opt, str):
                    opt_text = opt
                    opt_box = None
                elif isinstance(opt, dict):
                    opt_text = opt.get("text", "")
                    opt_box = opt.get("box")
                else:
                    print(f"  [WARN] Skipping invalid option type: {type(opt).__name__}")
                    continue

                label_prefix = f"{chr(65 + i)}) "
                cleaned_text = self.clean_option_text(opt_text)

                if has_imgs:
                    final_q["options"].append(label_prefix + cleaned_text)

                    if opt_box:
                        save_path = os.path.join(self.set_dir, fname)
                        if not extract_raw_image_if_exists(self.doc, page, opt_box, save_path):
                            process_and_save_crop(cv2_img, opt_box, save_path)

                        if os.path.exists(save_path):
                            final_q["image_option"].append(fname)
                        else:
                            print(f"  [WARN] Option image failed Q{q_num} Opt{i + 1}. Fallback to text.")
                            final_q["image_option"].append(label_prefix + cleaned_text)
                    else:
                        # Text-only option inside a mixed question —
                        # image_option gets the text so both lists stay
                        # the same length (parallel arrays).
                        final_q["image_option"].append(label_prefix + cleaned_text)
                else:
                    # Text-only mode → populate options[]
                    final_q["options"].append(label_prefix + cleaned_text)

            # Check for duplicate Q number
            existing_q_nums = {q["question_no"] for q in self.final_json}
            if q_num in existing_q_nums:
                print(f"  [WARN] Duplicate Q{q_num} detected. Skipping.")
                continue

            # Use the new validation method
            self.add_question_to_json(final_q)
            print(f"  Saved Q{q_num}")

    # ----------------------------------------------------------
    # POST-PROCESSING: Merge orphaned match-list options
    # ----------------------------------------------------------
    def merge_orphaned_matchlist_options(self):
        """
        Fallback safety net: If Gemini didn't use CONT for match-list options,
        we detect and merge them programmatically.
        
        Pattern:
          Question N: has image_question (table), but options: []
          Question N+1: has options but no q_text
          → N+1's options belong to N
        """
        print("[INFO] Post-processing: Checking for orphaned match-list options...")
        
        merged_count = 0
        questions_to_remove = []
        
        for i in range(len(self.final_json) - 1):
            curr_q = self.final_json[i]
            next_q = self.final_json[i + 1]
            
            # Pattern detection
            has_table_no_opts = (curr_q.get("image_question") and 
                                len(curr_q.get("options", [])) == 0)
            
            has_opts_no_text = (len(next_q.get("options", [])) > 0 and 
                               not next_q.get("question_txt") and
                               not next_q.get("image_question"))
            
            if has_table_no_opts and has_opts_no_text:
                print(f"  [ORPHAN MERGE] Q{curr_q['question_no']} + orphaned options from Q{next_q['question_no']}")
                
                # Transfer options
                curr_q["options"] = next_q["options"]
                curr_q["image_option"] = next_q.get("image_option", [])
                
                # Mark next_q for removal
                questions_to_remove.append(i + 1)
                merged_count += 1
        
        # Remove orphaned "questions" (they were just option blocks)
        for idx in reversed(questions_to_remove):
            removed_q = self.final_json.pop(idx)
            print(f"  [REMOVED] Orphan Q{removed_q['question_no']} (merged into previous)")
        
        if merged_count > 0:
            print(f"[SUCCESS] Merged {merged_count} orphaned match-list option blocks.")
        else:
            print("[OK] No orphaned match-list options found.")

    # ----------------------------------------------------------
    # Final save — split by subject, apply answer keys, write JSON
    # ----------------------------------------------------------
    def save(self):
        if not self.final_json:
            print("[WARN] No questions to save.")
            return

        # Report numerical question counts before saving
        print("\n" + "="*50)
        print(f"NUMERICAL QUESTION SUMMARY for Year {self.year}")
        print(f"Allowed per subject: {self.numerical_limit_per_subject}")
        print("-"*50)
        for subject, count in self.numerical_count.items():
            status = "✓ OK" if count <= self.numerical_limit_per_subject else "✗ EXCEEDED"
            print(f"{subject}: {count}/{self.numerical_limit_per_subject} {status}")
        print("="*50 + "\n")

        split_data = {"PHY": [], "CHEM": [], "MATH": []}

        for q in self.final_json:
            # ENFORCE QID-based subject (JEE fixed structure)
            # 2025 (75Q): Q1-25=Maths, Q26-50=Physics, Q51-75=Chemistry
            # 2024 (90Q): Q1-30=Physics, Q31-60=Chemistry, Q61-90=Maths
            qid = int(q["question_no"])
            year = self.metadata.get("year", 2025)
            
            if year == 2020: # 75-question format (P-C-M)
                if 1 <= qid <= 25:
                    sub_code, std_subject = "PHY", "physics"
                elif 26 <= qid <= 50:
                    sub_code, std_subject = "CHEM", "chemistry"
                else:
                    sub_code, std_subject = "MATH", "mathematics"
            elif year >= 2025:  # 75-question format
                if 1 <= qid <= 25:
                    sub_code, std_subject = "MATH", "mathematics"
                elif 26 <= qid <= 50:
                    sub_code, std_subject = "PHY", "physics"
                else:  # 51-75
                    sub_code, std_subject = "CHEM", "chemistry"
            else:  # 2024 and earlier (excluding 2020): 90-question format
                if 1 <= qid <= 30:
                    sub_code, std_subject = "PHY", "physics"
                elif 31 <= qid <= 60:
                    sub_code, std_subject = "CHEM", "chemistry"
                else:  # 61-90
                    sub_code, std_subject = "MATH", "mathematics"

            # Inner set name
            global_idx = self.metadata.get("set_global", 0)
            inner_set_name = f"{sub_code}_SET_{global_idx}"

            q["set"]     = inner_set_name
            q["subject"] = std_subject

            # Preserve msq in save() as well.
            raw_qt = q.get("question_type", "mcq").lower()
            if raw_qt == "numerical":
                q["question_type"] = "Numerical"
            elif raw_qt == "msq":
                q["question_type"] = "msq"
            else:
                q["question_type"] = "mcq"

            if self.answer_key_map:
                ans = self.answer_key_map.get(str(q["question_no"]))
                if ans is not None:
                    q["answer"] = ans

            split_data[sub_code].append(q)

        # Outer folder
        year      = self.metadata.get("year", "0000")
        local_idx = self.metadata.get("set_local", 0)
        outer_folder_name = f"JEE_{year}_SET_{local_idx}"
        outer_folder_path = os.path.join(OUTPUT_ROOT, "JEE", "OUTPUT", outer_folder_name)

        for sub_code, questions in split_data.items():
            if not questions:
                continue

            target_set_name = questions[0]["set"]
            set_output_dir  = os.path.join(outer_folder_path, target_set_name)
            os.makedirs(set_output_dir, exist_ok=True)

            # Move images 
            for q in questions:
                if q.get("image_question"):
                    src = os.path.join(self.set_dir, q["image_question"])
                    if os.path.exists(src):
                        shutil.copy2(src, os.path.join(set_output_dir, q["image_question"]))

                for item in q.get("image_option", []):
                    # item is a filename only if it ends with .png
                    if item.endswith(".png"):
                        src = os.path.join(self.set_dir, item)
                        if os.path.exists(src):
                            shutil.copy2(src, os.path.join(set_output_dir, item))

            # Write JSON
            fname     = f"{target_set_name}.json"
            save_path = os.path.join(set_output_dir, fname)

            questions.sort(key=lambda x: x["question_no"])
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(questions, f, indent=4, ensure_ascii=False)
            print(f"[SUCCESS] Saved {len(questions)} Qs to {save_path}")

    # ----------------------------------------------------------
    # Main run loop
    # ----------------------------------------------------------
    def run(self):
        for i in range(len(self.doc)):
            snapshot_before = len(self.final_json)

            try:
                self.process_page(i)
            except Exception as e:
                print(f"[ERROR] Page {i + 1} crashed: {e}")
                import traceback
                traceback.print_exc()

            # --- Jump detection (after page processed) ---
            if len(self.final_json) > snapshot_before:
                # At least one question was added on this page
                first_new = self.final_json[snapshot_before]
                first_q_no = int(first_new["question_no"])

                # Compare against the question BEFORE this page
                if snapshot_before > 0:
                    last_prev_q_no = int(self.final_json[snapshot_before - 1]["question_no"])
                    if first_q_no > last_prev_q_no + 1:
                        missing = first_q_no - last_prev_q_no - 1
                        print(f"  [ALERT] QUESTION JUMP! Missing {missing} Qs "
                              f"(expected Q{last_prev_q_no + 1}, got Q{first_q_no}).")

            time.sleep(API_DELAY_SECONDS)

        # POST-PROCESSING: Merge orphaned match-list options
        self.merge_orphaned_matchlist_options()

        # Crash-safe save
        print("[INFO] Saving progress...")
        self.save()

# =======================
# 4. CONFIG GENERATOR  
# =======================

def generate_jee_configs(years):
    configs = []

    sorted_years = sorted(years, reverse=True)   # newest first for global counter
    global_set_counter = 0

    for year in sorted_years:
        year_path = os.path.join(INPUT_FOLDER, str(year))
        if not os.path.exists(year_path):
            print(f"[WARN] Folder for {year} not found: {year_path}")
            continue

        pdf_files = [f for f in os.listdir(year_path) if f.lower().endswith(".pdf")]

        # Date-aware sort: (Year, Month, Day, Shift)
        def sort_key(fname):
            # Try standard format with Shift
            match = re.search(r"JEE_Main_(\d{4})_(\d{2})_([A-Za-z]{3})_Shift_(\d)", fname)
            if match:
                y, d, m, s = match.groups()
                return (int(y), MONTH_MAP.get(m, 0), int(d), int(s))
            
            # Format: JEE_Main_2018_08_Apr.pdf (Date, no shift)
            match = re.search(r"JEE_Main_(\d{4})_(\d{2})_([A-Za-z]{3})", fname)
            if match:
                y, d, m = match.groups()
                return (int(y), MONTH_MAP.get(m, 0), int(d), 0)

            # Format: JEE_Main_2018_16_Tru_Online.pdf (Year, Day, Weird Tag)
            match = re.search(r"JEE_Main_(\d{4})_(\d{2})_", fname)
            if match:
                y, d = match.groups()
                return (int(y), 0, int(d), 0)

            # Fallback: Just sort by year (if found) or filename
            match = re.search(r"JEE_Main_(\d{4})", fname)
            if match:
                return (int(match.group(1)), 0, 0, 0)
                
            return (0, 0, 0, 0)

        pdf_files.sort(key=sort_key)

        for i, pdf_file in enumerate(pdf_files):
            local_idx        = i + 1
            global_set_counter += 1

            ref_name       = os.path.splitext(pdf_file)[0].replace(" ", "_")
            unique_set_id  = f"JEE_{year}_SET_{local_idx}"

            config = {
                "file": os.path.join(year_path, pdf_file),   # full path
                "metadata": {
                    "set":        unique_set_id,
                    "year":       year,
                    "set_local":  local_idx,
                    "set_global": global_set_counter,
                    "grade":      12,
                    "reference":  ref_name,
                    "prepmode":   "JEE",
                    "subject":    "PCM"
                }
            }
            configs.append(config)

    return configs

# ==========================================
# 4. AUDIT UTILS
# ==========================================
def audit_existing_json_for_numerical_limits(json_path, year):
    """
    Audits an existing JSON file to check if it complies with numerical question limits.
    """
    limit = get_numerical_limit_for_year(year)
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        numerical_count = sum(1 for q in questions if q.get("question_type") == "Numerical")
        
        status = "PASS" if numerical_count <= limit else "FAIL"
        print(f"[AUDIT] {json_path}")
        print(f"        Year: {year}, Limit: {limit}, Found: {numerical_count} [{status}]")
        
        return numerical_count <= limit
    except Exception as e:
        print(f"[ERROR] Could not audit {json_path}: {e}")
        return None

# ==========================================
# 5. ENTRY POINT
# ==========================================

def initialize_gemini():
    global client
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in .env")
    client = genai.Client(api_key=GEMINI_API_KEY)
    print("[INFO] Gemini API initialized.")

if __name__ == "__main__":
    initialize_gemini()

    # Process ALL years down to 2012
    PROCESS_YEARS = [
        2025, 2024, 2023, 2022, 2021, 2020, 2019,
        2018, 2017, 2016, 2015, 2014, 2013, 2012
    ]
    PDF_CONFIGS = generate_jee_configs(PROCESS_YEARS)

    # Remove limit to process all papers
    # PDF_CONFIGS = PDF_CONFIGS[:2]  # Uncomment to limit for testing

    print(f"[INFO] Processing {len(PDF_CONFIGS)} sets")

    for config in PDF_CONFIGS:
        pdf_file = config["file"]
        meta     = config["metadata"]
        
        # Check if already processed (skip if complete)
        set_name = meta["set"]
        year = meta.get("year", 2025)
        set_local = meta.get("set_local", 1)
        outer_folder = f"JEE_{year}_SET_{set_local}"
        output_base = os.path.join("ROOT", "JEE", "OUTPUT", outer_folder)
        
        # Check if all 3 subject JSONs exist
        all_exist = True
        for sub in ["PHY", "CHEM", "MATH"]:
            sub_folder = f"{sub}_SET_{meta.get('set_global', set_local)}"
            json_path = os.path.join(output_base, sub_folder, f"{sub_folder}.json")
            if not os.path.exists(json_path):
                all_exist = False
                break
        
        if all_exist:
            print(f"\n[SKIP] {set_name} already processed. Skipping...")
            continue

        print(f"\n{'=' * 50}")
        print(f"PROCESSING: {pdf_file}")
        print(f"METADATA:   {meta}")
        print(f"{'=' * 50}\n")

        try:
            extractor = DirectExtractor(pdf_file, metadata=meta)
            extractor.run()
        except Exception as e:
            print(f"[ERROR] Failed {pdf_file}: {e}")
            import traceback
            traceback.print_exc()