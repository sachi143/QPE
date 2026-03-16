"""
NEET Paper Batch Processing Script

Processes all NEET papers systematically with proper chronological
set naming and global ID assignment similar to JEE processing.

NEET Exam Structure:
- 2025: 180 compulsory questions (45 Phy + 45 Chem + 90 Bio)
- 2021-2024: 200 questions (35 compulsory + 10 optional from 15 per subject)
- Pre-2021: 180 questions (45 + 45 + 90)

Subject Order: Physics → Chemistry → Biology (Botany + Zoology)
"""

import os
import sys
import re

# Resolve project root (2 levels up from scripts/NEET/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Add script directory to path for local imports
sys.path.insert(0, SCRIPT_DIR)

from geminiv3 import DirectExtractor, initialize_gemini

# Directories (resolved from project root)
INPUT_FOLDER = os.path.join(PROJECT_ROOT, "raw_pdfs", "NEET_PYQPs")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "ROOT")

def extract_year_from_filename(filename):
    """Extract year from NEET PDF filename"""
    # Try pattern: NEET_YYYY
    match = re.search(r'NEET_(\d{4})', filename)
    if match:
        return int(match.group(1))
    
    # Try pattern: NEET_YYYY-YY (for 2021-22)
    match = re.search(r'NEET_(\d{4})-\d{2}', filename)
    if match:
        return int(match.group(1))
    
    return None

def get_expected_question_count(year):
    """
    Returns expected total question count based on year
    
    2021-2024: 200 questions (attempted: 180)
    Other years: 180 questions
    """
    if 2021 <= year <= 2024:
        return 200  # Total questions in paper
    else:
        return 180  # Compulsory questions

def generate_neet_configs():
    """
    Generate processing configurations for all NEET PDFs.
    Returns list of configs sorted chronologically.
    """
    if not os.path.exists(INPUT_FOLDER):
        print(f"[ERROR] Input folder not found: {INPUT_FOLDER}")
        return []
    
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith('.pdf')]
    
    # Group files by year
    year_files = {}
    sample_files = []  # For Oswaal/Sample papers without clear year
    
    for fname in files:
        # Check for Oswaal/Sample papers (NEET_O1, NEET_O2, etc.)
        if re.match(r'NEET_O\d+\.pdf', fname, re.IGNORECASE):
            sample_files.append(fname)
            continue
            
        year = extract_year_from_filename(fname)
        if year:
            if year not in year_files:
                year_files[year] = []
            year_files[year].append(fname)
    
    # Sort years
    sorted_years = sorted(year_files.keys())
    
    # Generate configs
    configs = []
    global_id = 1
    
    for year in sorted_years:
        # Sort files within year alphabetically for consistency
        sorted_files = sorted(year_files[year])
        
        for local_idx, fname in enumerate(sorted_files, start=1):
            set_id = f"NEET_{year}_SET_{local_idx}"
            ref_name = os.path.splitext(fname)[0].replace(" ", "_")
            
            config = {
                "file": os.path.join(INPUT_FOLDER, fname),
                "metadata": {
                    "set": set_id,
                    "year": year,
                    "set_local": local_idx,
                    "set_global": global_id,
                    "grade": 12,
                    "reference": ref_name,
                    "prepmode": "NEET",
                    "subject": "PCB",  # Physics, Chemistry, Biology
                    "expected_questions": get_expected_question_count(year)
                }
            }
            
            configs.append(config)
            global_id += 1
    
    # Add sample/Oswaal papers at the end
    if sample_files:
        sorted_sample_files = sorted(sample_files)
        for local_idx, fname in enumerate(sorted_sample_files, start=1):
            # Extract number from NEET_O1, NEET_O2 pattern
            match = re.search(r'O(\d+)', fname)
            set_num = int(match.group(1)) if match else local_idx
            
            set_id = f"NEET_Sample_SET_{set_num}"
            ref_name = os.path.splitext(fname)[0].replace(" ", "_")
            
            config = {
                "file": os.path.join(INPUT_FOLDER, fname),
                "metadata": {
                    "set": set_id,
                    "year": 2025,  # Assume recent for sample papers
                    "set_local": set_num,
                    "set_global": global_id,
                    "grade": 12,
                    "reference": ref_name,
                    "prepmode": "NEET",
                    "subject": "PCB",
                    "expected_questions": 180  # Standard NEET format
                }
            }
            
            configs.append(config)
            global_id += 1
    
    return configs

def run_batch_processing(skip_existing=True):
    """
    Process all NEET papers in chronological order.
    
    Args:
        skip_existing: If True, skip sets that already have complete output
    """
    print("="*60)
    print("NEET BATCH PROCESSING")
    print("="*60)
    
    # Initialize Gemini
    initialize_gemini()
    
    # Generate configs
    configs = generate_neet_configs()
    
    if not configs:
        print("[ERROR] No valid NEET PDFs found.")
        return
    
    print(f"\nFound {len(configs)} NEET papers to process")
    print(f"Year range: {configs[0]['metadata']['year']} - {configs[-1]['metadata']['year']}")
    print()
    
    # Process each paper
    success_count = 0
    skip_count = 0
    
    for config in configs:
        meta = config['metadata']
        set_name = meta['set']
        year = meta['year']
        set_local = meta['set_local']
        
        # Check if already processed (skip if complete)
        if skip_existing:
            outer_folder = f"NEET_{year}_SET_{set_local}"
            output_base = os.path.join(OUTPUT_ROOT, "NEET", "OUTPUT", outer_folder)
            
            # Check if JSON exists (simple check)
            json_path = os.path.join(output_base, f"{outer_folder}.json")
            if os.path.exists(json_path):
                print(f"\n[SKIP] {set_name} already processed.")
                skip_count += 1
                continue
        
        print(f"\n{'='*60}")
        print(f"PROCESSING: {config['file']}")
        print(f"SET: {set_name} (Global ID: {meta['set_global']})")
        print(f"YEAR: {year} | Expected Questions: {meta['expected_questions']}")
        print(f"{'='*60}\n")
        
        try:
            extractor = DirectExtractor(config['file'], metadata=meta)
            extractor.run()
            success_count += 1
            print(f"[SUCCESS] {set_name} completed!")
        except Exception as e:
            print(f"[ERROR] Failed {set_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Processed: {success_count}")
    print(f"Skipped: {skip_count}")
    print(f"Total: {len(configs)}")
    print()

if __name__ == "__main__":
    run_batch_processing(skip_existing=True)
