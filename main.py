# --- EXAMPLE USAGE ---
import json

from synth_data_architect import generate_synthetic_dataset

if __name__ == "__main__":
    # Example Parameters
    N_ROWS = 1000000
    JSON_FILE = "tax_rules.json"
    SCHEMA = "tax_return_id, taxpayer_id, tax_year, form_type, filing_status, zip_code, state, city, gross_income, taxable_income, total_deductions, tax_credit_amount, tax_liability, withholding_amount, refund_amount, filing_date, payment_date, due_date, audit_risk_score, original_return_id, preparer_id"
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    RULES = data

    # Run
    df = generate_synthetic_dataset(N_ROWS, SCHEMA, RULES)