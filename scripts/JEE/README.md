# JEE Scripts

## Current Files

### Core Scripts

#### `geminiv1.py`
Main extraction engine for JEE papers. Processes PDFs using Gemini Vision API.

**Features**:
- Handles 2012-2025 JEE papers (153 sets)
- Year-specific numerical question limits
- Answer key integration
- Image extraction and cleanup
- Logo rejection and watermark removal

**Usage**:
```bash
cd scripts/JEE
python geminiv1.py
```

**Note**: This is the primary extraction script. Configure PDFs in the script before running.

---

### Verification Scripts

#### `verify_jee.py` (Recommended)
Comprehensive verification with HTML report generation.

**Features**:
- Verifies all 153 JEE sets
- Checks question continuity (1-90 or 1-75)
- Subject distribution validation (Phy/Chem/Math)
- Generates interactive HTML reports with images

**Usage**:
```bash
cd scripts/JEE
python verify_jee.py
```

**Output**: Creates `VERIFICATION_REPORT.html` in each set folder.

---

#### `check_output.py`
Quick completeness check for extracted sets.

**Features**:
- Lists all sets with question counts
- Identifies missing questions
- Simple text-based output

**Usage**:
```bash
cd scripts/JEE
python check_output.py
```

---

## Output Structure

```
ROOT/JEE/OUTPUT/
├── JEE_2012_SET_1/
│   ├── PHY_SET_1/
│   │   ├── PHY_SET_1.json
│   │   └── ...images...
│   ├── CHEM_SET_1/
│   │   ├── CHEM_SET_1.json
│   │   └── ...images...
│   └── MATH_SET_1/
│       ├── MATH_SET_1.json
│       └── ...images...
├── JEE_2012_SET_2/
...
└── JEE_2025_SET_12/
```

## Dataset Status

- **Total Sets**: 153 (Global IDs 1-153)
- **Years Covered**: 2012-2025
- **Total Questions**: ~12,690
- **Question Distribution**: 
  - 2012-2019: 90 questions/set
  - 2020: 75 questions/set
  - 2021-2024: 90 questions/set
  - 2025: 75 questions/set

## Current Status

All 153 JEE sets have been processed and verified. Use `verify_jee.py` to generate fresh verification reports anytime.

## Additional Documentation

See [`set_mapping_complete.md`](set_mapping_complete.md) for the complete mapping of PDFs to set names and global IDs.

