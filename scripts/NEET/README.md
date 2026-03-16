# NEET Scripts

## Main Scripts

### `geminiv3.py`
Main extraction engine for NEET papers. Processes PDFs using Gemini Vision API.

**Features**:
- Handles 2013-2025 NEET papers
- Logo rejection (PW, Oswaal)
- Inverted image correction
- Biology-focused extraction (PCB)

**Usage**:
```bash
python geminiv3.py
```

### `batch_neet.py`
Batch processing for all 49 NEET papers with chronological set naming.

**Usage**:
```bash
python batch_neet.py
```

**Output**: Processes all 49 sets sequentially with global ID assignment.

### `check_neet_output.py`
Verifies completeness of extracted NEET sets (180 questions per set).

**Usage**:
```bash
python check_neet_output.py
```

## NEET Exam Structure

| Year | Total Questions | Physics | Chemistry | Biology |
|------|----------------|---------|-----------|---------|
| 2025 | 180 | 45 | 45 | 90 |
| 2021-2024 | 200 (180 attempted) | 45 | 45 | 90 |
| Pre-2021 | 180 | 45 | 45 | 90 |

## Output Structure
```
ROOT/NEET/OUTPUT/
├── NEET_2013_SET_1/
├── NEET_2014_SET_1/
...
├── NEET_2025_SET_2/
├── NEET_Sample_SET_1/  # Oswaal sample papers
└── NEET_Sample_SET_2/
```

## Total Sets
49 sets (47 regular + 2 samples, Global IDs 1-49)
