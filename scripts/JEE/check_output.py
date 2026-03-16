import json
import os

# Resolve project root (2 levels up from scripts/JEE/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
OUTPUT_ROOT_DIR = os.path.join(PROJECT_ROOT, "ROOT", "JEE", "OUTPUT")

def check_set(set_folder):
    print(f"\n=== {set_folder} ===")
    subjects = ["PHY", "CHEM", "MATH"]
    all_qnums = []
    
    base_path = os.path.join(OUTPUT_ROOT_DIR, set_folder)
    
    for sub in subjects:
        # Find actual folder - could be PHY_SET_1 or PHY_SET_11 etc
        found = False
        for item in os.listdir(base_path):
            if item.startswith(f"{sub}_SET_") and os.path.isdir(os.path.join(base_path, item)):
                folder_name = item
                path = os.path.join(base_path, folder_name, f"{folder_name}.json")
                if os.path.exists(path):
                    data = json.load(open(path))
                    qnums = sorted([q["question_no"] for q in data])
                    all_qnums.extend(qnums)
                    print(f"  {sub}: {len(data)} Qs | Q{qnums[0]}-Q{qnums[-1]}")
                    found = True
                    break
        if not found:
            print(f"  {sub}: NOT FOUND")
    
    # Check for missing - detect year from folder name
    year = int(set_folder.split("_")[1]) if "_" in set_folder else 2025
    if year >= 2025 or year == 2020:
        total_qs = 75
    else:
        total_qs = 90
    
    all_qnums = sorted(set(all_qnums))
    expected = set(range(1, total_qs + 1))
    missing = expected - set(all_qnums)
    if missing:
        print(f"  ⚠️ MISSING: {sorted(missing)}")
        return False
    else:
        print(f"  ✅ All {total_qs} questions present!")
        return True

# Check all sets in OUTPUT folder
output_dir = OUTPUT_ROOT_DIR
if os.path.exists(output_dir):
    sets = sorted([d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))])
else:
    print(f"Output directory not found: {output_dir}")
    sets = []

print(f"Found {len(sets)} sets to verify\n")

complete = 0
incomplete = 0
for s in sets:
    if check_set(s):
        complete += 1
    else:
        incomplete += 1

print(f"\n{'='*50}")
print(f"SUMMARY: {complete} complete, {incomplete} incomplete")
