# --- EXAMPLE USAGE ---
import json

from utils.main_functions import generate_synthetic_dataset

if __name__ == "__main__":
    # Example Parameters
    N_ROWS = 1000000
    JSON_FILE = "dataset_generated/stress_test_general/hospital_rules.json"

    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    RULES = data

    # Run
    df = generate_synthetic_dataset(num_rows=N_ROWS,
                                    rules=RULES)