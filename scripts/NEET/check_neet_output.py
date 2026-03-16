"""
NEET Output Verification Script

Verifies extracted NEET question sets for completeness.

Expected structure:
- 2025: 180 questions (45 Phy + 45 Chem + 90 Bio)
- 2021-2024: 180 attempted (out of 200 total)
- Pre-2021: 180 questions

Question numbering:
Q1-Q45: Physics
Q46-Q90: Chemistry  
Q91-Q180 (or Q200): Biology (Botany + Zoology)
"""

import os
import json

# Resolve project root (2 levels up from scripts/NEET/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "ROOT", "NEET", "OUTPUT")

def get_expected_count(year):
    """Returns expected question count based on yearfor NEET"""
    if 2021 <= year <= 2024:
        return 180  # Attempted questions (out of 200 total)
    else:
        return 180  # Standard format

def check_set(set_folder):
    """Verify a single NEET set for completeness"""
    # Extract year from folder name
    parts = set_folder.split("_")
    if len(parts) >= 2:
        try:
            year = int(parts[1])
        except:
            year = 2025
    else:
        year = 2025
    
    total_expected = get_expected_count(year)
    
    # Load JSON file
    json_path = os.path.join(OUTPUT_ROOT, set_folder, f"{set_folder}.json")
    
    if not os.path.exists(json_path):
        return f"  ❌ JSON not found"
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract question numbers
        q_numbers = []
        for item in data:
            if 'question_no' in item:
                q_numbers.append(item['question_no'])
        
        # Sort and count
        q_numbers.sort()
        total_found = len(q_numbers)
        
        # Check for gaps
        expected_range = range(1, total_expected + 1)
        missing = [q for q in expected_range if q not in q_numbers]
        
        # Subject breakdown (approximate based on standard ranges)
        phy_qs = [q for q in q_numbers if 1 <= q <= 45]
        chem_qs = [q for q in q_numbers if 46 <= q <= 90]
        bio_qs = [q for q in q_numbers if 91 <= q <= total_expected]
        
        if missing:
            return (f"  PHY: {len(phy_qs)} Qs | Q{min(phy_qs) if phy_qs else '?'}-Q{max(phy_qs) if phy_qs else '?'}\n"
                    f"  CHEM: {len(chem_qs)} Qs | Q{min(chem_qs) if chem_qs else '?'}-Q{max(chem_qs) if chem_qs else '?'}\n"
                    f"  BIO: {len(bio_qs)} Qs | Q{min(bio_qs) if bio_qs else '?'}-Q{max(bio_qs) if bio_qs else '?'}\n"
                    f"  ⚠️ MISSING: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
        else:
            return (f"  PHY: {len(phy_qs)} Qs | Q{min(phy_qs)}-Q{max(phy_qs)}\n"
                    f"  CHEM: {len(chem_qs)} Qs | Q{min(chem_qs)}-Q{max(chem_qs)}\n"
                    f"  BIO: {len(bio_qs)} Qs | Q{min(bio_qs)}-Q{max(bio_qs)}\n"
                    f"  ✅ All {total_expected} questions present!")
    
    except Exception as e:
        return f"  ❌ Error reading JSON: {e}"

def main():
    """Verify all NEET sets"""
    if not os.path.exists(OUTPUT_ROOT):
        print(f"Output folder not found: {OUTPUT_ROOT}")
        return
    
    sets = sorted([d for d in os.listdir(OUTPUT_ROOT) 
                   if os.path.isdir(os.path.join(OUTPUT_ROOT, d)) and d.startswith("NEET_")])
    
    print(f"Found {len(sets)} sets to verify\n")
    
    complete_count = 0
    incomplete_count = 0
    
    for set_folder in sets:
        print(f"\n=== {set_folder} ===")
        result = check_set(set_folder)
        print(result)
        
        if "✅" in result:
            complete_count += 1
        else:
            incomplete_count += 1
    
    print(f"\n{'='*50}")
    print(f"SUMMARY: {complete_count} complete, {incomplete_count} incomplete")

if __name__ == "__main__":
    main()
