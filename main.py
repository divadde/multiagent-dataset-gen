# --- EXAMPLE USAGE ---
import json

from synt_data_architect import generate_synthetic_dataset

if __name__ == "__main__":
    # Example Parameters
    N_ROWS = 1000000
    SCHEMA = "visit_id, patient_id, patient_name, date_of_birth, admission_date, discharge_date, department_id, department_name, doctor_id, diagnosis_code, treatment_cost, insurance_coverage, visit_status, room_number, bed_type, emergency_contact, blood_type"
    with open('example_rules.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    RULES = data

    # Run
    df = generate_synthetic_dataset(N_ROWS, SCHEMA, RULES)