# Question Paper Extraction Project

A Python-based pipeline for automated extraction of questions, options, and answers from JEE and NEET exam papers using Google Generative AI (Gemini Vision API).

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
├── scripts/
│   ├── JEE/           # JEE-specific scripts
│   │   ├── geminiv1.py
│   │   ├── verify_jee.py
│   │   └── check_output.py
│   │
│   └── NEET/          # NEET-specific scripts
│       ├── geminiv3.py
│       ├── batch_neet.py
│       └── check_neet_output.py
│
├── documentation/     # Client documentation
├── raw_pdfs/          # Source PDFs
│   ├── JEE_PYQPs/
│   └── NEET_PYQPs/
│
└── ROOT/              # Extracted output
    ├── JEE/OUTPUT/
    └── NEET/OUTPUT/
```

## Quick Start

### JEE Processing (153 sets)
```bash
cd scripts/JEE
python geminiv1.py          # Main extraction
python check_output.py      # Verify completeness
```

### NEET Processing (49 sets)
```bash
cd scripts/NEET
python batch_neet.py        # Batch process all sets
python check_neet_output.py # Verify completeness
```

## Statistics

| Exam | Years | Total Sets | Total Questions |
|------|-------|-----------|----------------|
| **JEE** | 2012-2025 | 153 | ~12,690 |
| **NEET** | 2013-2025 | 49 | ~8,820 |

## Documentation

- [JEE Scripts README](scripts/JEE/README.md)
- [NEET Scripts README](scripts/NEET/README.md)
