# --- EXAMPLE USAGE ---
import json

from utils.analyze_rule_dependencies import visualize_attribute_dependencies
from utils.main_functions import generate_synthetic_dataset

if __name__ == "__main__":
    # Example Parameters
    N_ROWS = 1000000
    JSON_FILE = "dataset_generated/stress_test_fd/final_test.json"

    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    RULES = data

    visualize_attribute_dependencies(RULES, output_path="dependency_graph_check.png")

    # Run
    df = generate_synthetic_dataset(num_rows=N_ROWS,
                                    rules=RULES)