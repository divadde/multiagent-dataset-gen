# --- EXAMPLE USAGE ---
import json

from synth_data_architect import generate_synthetic_dataset

if __name__ == "__main__":
    # Example Parameters
    N_ROWS = 1000000
    JSON_FILE = "dataset_generated/stress_test_general/hospital_rules.json"
    SCHEMA = "Infer from the rules \"target columns\". [No assertion needed to verify the schema] "
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    RULES = data

    # Run
    df = generate_synthetic_dataset(N_ROWS, SCHEMA, RULES)