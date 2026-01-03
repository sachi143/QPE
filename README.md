# JEE Exam Paper Question Extraction

A Python-based pipeline for automated extraction of questions, options, and answers from JEE exam papers using Google Generative AI (Gemini Vision API).

## Overview

This project processes JEE Main and NEET exam papers (PDFs) to extract:
- Questions with full text and formatting
- Multiple choice options (text and image-based)
- Question diagrams and figures
- Answer keys and mappings
- Structured metadata for each question

## Project Structure

```
touch_v2/
├── venv/                          # Local Python virtual environment
├── raw_pdfs/                      # Input exam papers
│   └── JEE Prev Year QPs/
│       ├── 2012-2025/            # Organized by year
├── ROOT/                          # Extraction output
│   └── JEE/Physics/               # Organized by subject
├── .env                           # API keys & configuration (git-ignored)
├── requirements.txt               # Python dependencies
├── gemini_jee.py                 # Main extraction pipeline
├── geminiv2.py                     # jee paper 
├── geminiv3.py                     # neet paper
├── .gitignore                     # Git ignore file
└── README.md                      # This file
```

## Setup

### 1. Install Dependencies

```bash
cd d:\evo11ve\touch_v2
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Key

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_api_key_here
```

### 3. Verify Environment

```bash
python -c "import cv2, fitz, google.generativeai; print('All dependencies installed!')"
```

## Usage

### Process Papers

Edit the PDF filename in `gemini_complete.py`:

```python
if __name__ == "__main__":
    PDF_FILE = "JEE_Main_2019_09_Jan_Shift_1.pdf"  # Change to desired paper
    if os.path.exists(os.path.join(INPUT_FOLDER, PDF_FILE)):
        DirectExtractor(PDF_FILE).run()
```

Then run:

```bash
python gemini_complete.py
```

Supported papers are located in:
- `raw_pdfs/JEE Prev Year QPs/2012-2025/` (organized by year)

## Output Structure

Each extracted paper generates:

```
ROOT/
└── JEE/Physics/[SET_NAME]/
    ├── [SET_NAME].json          # Main output with all questions
    ├── [SET_NAME]_Q01.png       # Question diagrams
    ├── [SET_NAME]_Q01_OPT_1.png # Option images (if image-based)
    └── ...
```

### JSON Schema

```json
[
  {
    "set": "JEE_2019_Jan09_Shift1",
    "grade": 12,
    "question_no": 1,
    "question_txt": "Full question text here...",
    "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
    "image_option": [],
    "answer": "A",
    "image_question": "JEE_2019_Jan09_Shift1_Q01.png",
    "reference": "JEE_MAIN_2019_JAN_09",
    "validity": "Valid",
    "prepmode": "JEE"
  }
]
```

## Key Features

- **Watermark Removal:** Automatic detection and removal of PDF watermarks
- **Ink Detection:** Precise extraction of question content using morphological operations
- **Vision API Integration:** Gemini 3 pro for intelligent question parsing
- **Flexible Options:** Handles both text and image-based multiple choice options
- **Answer Mapping:** Automatic matching of answers from answer key pages
- **Batch Processing:** Process multiple papers sequentially with progress tracking
- **Error Handling:** Graceful failure with detailed logging

## Technical Stack

- **Language:** Python 3.14
- **PDF Processing:** PyMuPDF (fitz)
- **Image Processing:** OpenCV (cv2), Pillow
- **Vision API:** Google Generative AI (Gemini Nano Banana Pro)
- **Environment:** Python virtual environment (venv)
- **Configuration:** python-dotenv

## Model Configuration

- **Model:** `gemini-3-pro-image-preview`
- **Vision Capability:** Full image analysis with OCR
- **DPI:** 300 DPI for PDF-to-image conversion
- **Response Format:** JSON with structured question/answer fields

## PDF Naming Convention

All input PDFs follow standardized naming:

```
JEE_Main_YYYY_DD_Mon_Shift_X.pdf
```

Examples:
- `JEE_Main_2019_08_Jan_Shift_1.pdf` (January 8, 2019, Shift 1)
- `JEE_Main_2019_09_Jan_Shift_2.pdf` (January 9, 2019, Shift 2)

## Processing Pipeline

1. **PDF Parsing:** Convert PDF pages to 300 DPI images using PyMuPDF
2. **Image Cleaning:** Remove watermarks, detect ink boundaries, tight crop
3. **Gemini Analysis:** Send images to Gemini API with question extraction prompt
4. **Text Cleaning:** Remove headers, footers, question numbers from extracted text
5. **Option Handling:** Distinguish between text-based and image-based options
6. **Answer Mapping:** Match extracted answers with answer key tables
7. **JSON Output:** Organize and save structured question data

## Known Issues & Limitations

- Empty question extraction on some papers (Gemini API model behavior variance)
- Image-based questions may require additional processing
- Large PDFs (100+ pages) may need chunking for optimal processing
- API rate limiting should be considered for batch jobs

## Contributing

For improvements or bug fixes, modify the DirectExtractor class or add new processing stages to `process_page()`.

## License

Internal Project
