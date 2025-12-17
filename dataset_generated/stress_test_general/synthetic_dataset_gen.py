import pandas as pd
import numpy as np
import random
import re

# Configuration
NUM_ROWS = 1_000_000
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Global maps used in validation
TRIAGE_WAIT_MAP = {'Red': 5, 'Orange': 10, 'Yellow': 30, 'Green': 60}
PROCEDURE_CAP_MAP = {}
PARENT_BUDGET_MAP = {}


def create_units_data(parent_count=200, children_per_parent=5):
    parent_ids = np.arange(1, parent_count + 1)
    parent_budgets = np.random.randint(5_000_000, 10_000_000, size=parent_count)
    parent_wings = np.random.choice(list("ABCDEFGH"), size=parent_count)
    parent_budget_map = dict(zip(parent_ids, parent_budgets))

    child_units = []
    child_parents = []
    child_budgets = []
    child_wings = []
    current_id = 1001
    for pid in parent_ids:
        for _ in range(children_per_parent):
            child_units.append(current_id)
            child_parents.append(pid)
            budget_val = np.random.randint(int(parent_budget_map[pid] * 0.3), int(parent_budget_map[pid] * 0.9))
            child_budgets.append(budget_val)
            child_wings.append(parent_wings[pid - 1])
            current_id += 1

    unit_parent_map = dict(zip(child_units, child_parents))
    unit_budget_map = dict(zip(child_units, child_budgets))
    unit_wing_map = dict(zip(child_units, child_wings))
    return child_units, unit_parent_map, unit_budget_map, parent_budget_map, unit_wing_map


def generate_dataset(num_rows):
    global PROCEDURE_CAP_MAP, PARENT_BUDGET_MAP

    child_units, unit_parent_map, unit_budget_map, parent_budget_map, unit_wing_map = create_units_data()
    PARENT_BUDGET_MAP = parent_budget_map

    df = pd.DataFrame({"admission_id": np.arange(1, num_rows + 1)})

    # Patient and staff identifiers
    df["patient_id"] = df["admission_id"].astype(str).str.zfill(7)
    df["staff_id"] = "S" + df["admission_id"].astype(str).str.zfill(7)
    df["doctor_id"] = "D" + df["admission_id"].astype(str).str.zfill(7)

    # Ward and hierarchy mappings
    num_wards = 500
    ward_ids = np.array([f"W{str(i).zfill(4)}" for i in range(1, num_wards + 1)])
    assigned_indices = (df["admission_id"].to_numpy() - 1) % num_wards
    df["assigned_ward_id"] = ward_ids[assigned_indices]

    head_nurses = np.array([f"HN{str(i).zfill(4)}" for i in range(1, num_wards + 1)])
    head_nurse_map = pd.Series(head_nurses, index=ward_ids)
    df["head_nurse_id"] = df["assigned_ward_id"].map(head_nurse_map)

    num_shift_managers = max(1, num_wards // 10)
    shift_managers = np.array([f"SM{str(i).zfill(3)}" for i in range(1, num_shift_managers + 1)])
    manager_idx = np.arange(num_wards) // 10
    shift_manager_map = pd.Series(shift_managers[manager_idx], index=head_nurses)
    df["shift_manager_id"] = df["head_nurse_id"].map(shift_manager_map)

    # Dates
    base_date = pd.Timestamp("2022-01-01")
    admission_offsets = np.random.randint(0, 365 * 2, size=num_rows)
    df["admission_date"] = base_date + pd.to_timedelta(admission_offsets, unit="D")

    start_offsets = np.random.randint(0, 3, size=num_rows)
    df["treatment_start_date"] = df["admission_date"] + pd.to_timedelta(start_offsets, unit="D")

    end_offsets = np.random.randint(1, 8, size=num_rows)
    df["treatment_end_date"] = df["treatment_start_date"] + pd.to_timedelta(end_offsets, unit="D")

    discharge_offsets = np.random.randint(0, 4, size=num_rows)
    df["discharge_date"] = df["treatment_end_date"] + pd.to_timedelta(discharge_offsets, unit="D")

    df["visit_date"] = df["admission_date"].dt.normalize()
    shift_hours = np.random.randint(0, 24, size=num_rows)
    df["shift_start_time"] = df["admission_date"].dt.normalize() + pd.to_timedelta(shift_hours, unit="H")

    # Admission type and triage
    admission_types = np.random.choice(["Emergency", "Routine", "Scheduled"], size=num_rows, p=[0.3, 0.4, 0.3])
    df["admission_type"] = admission_types

    triage_codes = np.array(["Red", "Orange", "Yellow", "Green"])
    triage_array = np.full(num_rows, "NA", dtype=object)
    emergency_mask = admission_types == "Emergency"
    triage_array[emergency_mask] = np.random.choice(triage_codes, size=emergency_mask.sum())
    df["triage_code"] = triage_array

    wait_times = np.full(num_rows, 120, dtype=int)
    for code, wait in TRIAGE_WAIT_MAP.items():
        mask = triage_array == code
        if mask.any():
            wait_times[mask] = wait
    df["max_wait_time"] = wait_times

    # Staff roles and licenses
    roles = np.random.choice(["Surgeon", "Nurse", "Therapist", "Resident"], size=num_rows, p=[0.2, 0.5, 0.2, 0.1])
    df["staff_role"] = roles
    surgery_license = np.full(num_rows, "", dtype=object)
    surg_mask = roles == "Surgeon"
    if surg_mask.any():
        surgery_license[surg_mask] = ("LIC" + df.loc[surg_mask, "staff_id"].str[1:]).values
    df["surgery_license_number"] = surgery_license

    # Insurance and procedures
    providers = np.random.choice(["Medicare", "Private", "None"], size=num_rows, p=[0.3, 0.5, 0.2])
    df["insurance_provider"] = providers
    procedure_codes = np.array([f"PRC{str(i).zfill(4)}" for i in range(1, 501)])
    df["procedure_code"] = np.random.choice(procedure_codes, size=num_rows)
    caps = np.random.randint(5000, 20000, size=len(procedure_codes))
    PROCEDURE_CAP_MAP = dict(zip(procedure_codes, caps))
    reimbursement = np.random.randint(1000, 15000, size=num_rows)
    mask_med = providers == "Medicare"
    if mask_med.any():
        reimbursement[mask_med] = pd.Series(df.loc[mask_med, "procedure_code"]).map(PROCEDURE_CAP_MAP).to_numpy()
    df["reimbursement_cap"] = reimbursement

    # Medical units and hierarchy
    unit_choices = np.random.choice(child_units, size=num_rows)
    unit_series = pd.Series(unit_choices)
    df["medical_unit_id"] = unit_choices
    df["parent_unit_id"] = unit_series.map(unit_parent_map).to_numpy()
    df["unit_budget"] = unit_series.map(unit_budget_map).to_numpy()
    df["parent_unit_budget"] = pd.Series(df["parent_unit_id"]).map(parent_budget_map).to_numpy()
    df["building_wing"] = unit_series.map(unit_wing_map).to_numpy()

    # Department and budgets
    num_departments = 500
    dept_ids = np.arange(1, num_departments + 1)
    dept_budgets_values = np.random.randint(5_000_000, 10_000_000, size=num_departments)
    department_budget_map = dict(zip(dept_ids, dept_budgets_values))
    department_name_map = {dept_id: f"Dept-{str(dept_id).zfill(4)}" for dept_id in dept_ids}
    df["department_id"] = ((df["admission_id"] - 1) % num_departments) + 1
    df["department_budget"] = pd.Series(df["department_id"]).map(department_budget_map).to_numpy()
    df["department_name"] = pd.Series(df["department_id"]).map(department_name_map).astype(str)
    df["staff_salaries"] = np.random.randint(100, 1000, size=num_rows)

    # Ward info
    df["ward_id"] = df["assigned_ward_id"]
    ward_total_beds_map = {ward_id: 2500 for ward_id in ward_ids}
    df["total_beds"] = df["ward_id"].map(ward_total_beds_map).to_numpy()
    df["active_patients"] = 1
    ward_types = np.random.choice(["ICU", "General", "Step-down"], size=num_rows, p=[0.2, 0.6, 0.2])
    df["ward_type"] = ward_types
    ratio_choices = np.random.choice(["1:3", "1:4", "1:5"], size=num_rows)
    df["nurse_patient_ratio"] = np.where(ward_types == "ICU", "1:1", ratio_choices)
    df["ward_code"] = df["assigned_ward_id"].str[1:]

    # Names and identifiers
    df["first_name"] = "First" + df["admission_id"].astype(str)
    df["last_name"] = "Last" + df["admission_id"].astype(str)
    dob_base = pd.Timestamp("1950-01-01")
    dob_offsets = np.random.randint(0, 365 * 70, size=num_rows)
    df["dob"] = (dob_base + pd.to_timedelta(dob_offsets, unit="D")).dt.date
    df["staff_last_name"] = "StaffLN" + df["admission_id"].astype(str)
    df["staff_initials"] = df["staff_last_name"].str[:2]

    df["diagnosis_code"] = "DX" + (df["admission_id"] % 1000).astype(str).str.zfill(3)
    df["treating_doctor_id"] = df["doctor_id"]
    df["short_diagnosis_code"] = "S" + (df["admission_id"] % 1000).astype(str).str.zfill(3)
    df["full_diagnosis_description"] = "Diagnosis description " + df["diagnosis_code"]

    # Patient status
    df["patient_status"] = "Discharged"

    # Contact and syntactic info
    df["email_address"] = "patient" + df["admission_id"].astype(str) + "@hospital.org"
    ssn_numbers = df["admission_id"].astype(str).str.zfill(9)
    df["social_security_number"] = ssn_numbers.str[:3] + "-" + ssn_numbers.str[3:5] + "-" + ssn_numbers.str[5:]
    df["internal_doc_link"] = "https://hospital-intranet/doc/" + df["admission_id"].astype(str)

    # Drug info
    df["drug_description"] = "Drug" + df["admission_id"].astype(str) + " 100mg"
    letters = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    lot_letters = np.random.choice(letters, size=(num_rows, 3))
    lot_numbers = np.random.randint(0, 10000, size=num_rows)
    df["drug_lot_number"] = ["LOT-" + "".join(lot_letters[i]) + "-" + str(lot_numbers[i]).zfill(4) for i in range(num_rows)]

    # DICOM IDs
    hex_strings = [f"{i:032x}" for i in range(1, num_rows + 1)]
    df["dicom_image_id"] = [h[:8] + "-" + h[8:12] + "-" + h[12:16] + "-" + h[16:20] + "-" + h[20:] for h in hex_strings]

    # Vitals
    df["body_temperature_celsius"] = np.random.uniform(36.0, 39.0, size=num_rows)
    df["heart_rate_bpm"] = np.random.randint(60, 100, size=num_rows)
    df["systolic_bp"] = np.random.randint(90, 140, size=num_rows)
    diastolic_vals = np.random.randint(60, 90, size=num_rows)
    df["blood_pressure"] = [f"{sys}/{dia}" for sys, dia in zip(df["systolic_bp"].to_numpy(), diastolic_vals)]

    # Financials
    total_invoice = np.random.randint(1000, 10000, size=num_rows)
    insurance_payout = np.random.randint(500, 9000, size=num_rows)
    insurance_payout = np.minimum(insurance_payout, total_invoice)
    patient_copay = total_invoice - insurance_payout
    df["total_invoice_amount"] = total_invoice
    df["insurance_payout"] = insurance_payout
    df["patient_copay"] = patient_copay

    # Anesthesia
    anesthesia = np.random.uniform(10, 50, size=num_rows)
    max_safe = anesthesia + np.random.uniform(1, 20, size=num_rows)
    df["anesthesia_dosage"] = anesthesia
    df["max_safe_dosage"] = max_safe

    # Consultations
    df["daily_consultations"] = np.random.randint(1, 6, size=num_rows)

    # Recovery and pain metrics
    rec1 = np.random.randint(0, 9, size=num_rows)
    rec7 = rec1 + np.random.randint(0, 3, size=num_rows)
    rec7 = np.clip(rec7, 0, 10)
    df["recovery_score_day_1"] = rec1
    df["recovery_score_day_7"] = rec7

    pain_adm = np.random.randint(5, 11, size=num_rows)
    pain_dis = pain_adm - np.random.randint(1, 5, size=num_rows)
    pain_dis = np.clip(pain_dis, 0, 10)
    df["pain_level_admission"] = pain_adm
    df["pain_level_discharge"] = pain_dis

    med1 = np.random.uniform(5, 50, size=num_rows)
    med2 = med1 + np.random.uniform(1, 20, size=num_rows)
    df["medication_dosage_step_1"] = med1
    df["medication_dosage_step_2"] = med2

    return df


def validate_data(df):
        assert len(df) == NUM_ROWS, "Row count mismatch"

        # Unique constraints
        assert df["admission_id"].is_unique, "admission_id not unique"
        assert not df[["patient_id", "visit_date"]].duplicated().any(), "patient_id + visit_date duplicates"
        assert not df[["staff_id", "shift_start_time"]].duplicated().any(), "staff_id + shift_start_time duplicates"

        # Temporal order
        assert (df["admission_date"] <= df["treatment_start_date"]).all()
        assert (df["treatment_start_date"] <= df["treatment_end_date"]).all()
        assert (df["treatment_end_date"] <= df["discharge_date"]).all()

        # Functional dependencies
        assert df.groupby("patient_id")["assigned_ward_id"].nunique().max() == 1
        assert df.groupby("assigned_ward_id")["head_nurse_id"].nunique().max() == 1
        assert df.groupby("head_nurse_id")["shift_manager_id"].nunique().max() == 1

        # CFDs
        em_mask = df["admission_type"] == "Emergency"
        if em_mask.any():
            triage_matches = df.loc[em_mask, "triage_code"].map(TRIAGE_WAIT_MAP).to_numpy()
            assert (df.loc[em_mask, "max_wait_time"].to_numpy() == triage_matches).all()
        surg_mask = df["staff_role"] == "Surgeon"
        if surg_mask.any():
            assert df.loc[surg_mask, "surgery_license_number"].ne("").all()
            assert df.loc[surg_mask].groupby("staff_id")["surgery_license_number"].nunique().max() == 1
        med_mask = df["insurance_provider"] == "Medicare"
        if med_mask.any():
            med_caps = df.loc[med_mask, "procedure_code"].map(PROCEDURE_CAP_MAP).to_numpy()
            assert (df.loc[med_mask, "reimbursement_cap"].to_numpy() == med_caps).all()

        # Intra-reference
        assert (df["medical_unit_id"] != df["parent_unit_id"]).all()
        parent_wings = df[["parent_unit_id", "building_wing"]].drop_duplicates()
        assert parent_wings.groupby("parent_unit_id")["building_wing"].nunique().max() == 1
        parent_budget_series = df["parent_unit_id"].map(PARENT_BUDGET_MAP)
        mask_parent = df["parent_unit_id"].isin(PARENT_BUDGET_MAP.keys())
        if mask_parent.any():
            assert (df.loc[mask_parent, "unit_budget"].to_numpy() < parent_budget_series.loc[mask_parent].to_numpy()).all()

        # Fuzzy duplicate avoidance
        normalized_drugs = df["drug_description"].str.replace(" ", "", regex=False).str.lower()
        assert not normalized_drugs.duplicated().any()
        name_key = (df["first_name"].str.lower() + df["last_name"].str.lower() + pd.Series(df["dob"]).astype(str))
        assert not name_key.duplicated().any()
        dept_unique = df[["department_id", "department_name"]].drop_duplicates("department_id")
        norm_dept = dept_unique["department_name"].str.replace(" ", "", regex=False).str.lower()
        assert not norm_dept.duplicated().any()

        # Implications and completeness
        assert df.loc[df["diagnosis_code"].notna(), "treating_doctor_id"].ne("").all()
        icu_mask = df["ward_type"] == "ICU"
        if icu_mask.any():
            assert (df.loc[icu_mask, "nurse_patient_ratio"] == "1:1").all()
        discharged_mask = df["patient_status"] == "Discharged"
        assert df.loc[discharged_mask, "discharge_date"].notna().all()

        # Syntax checks
        ssn_pattern = re.compile(r"^\d{3}-\d{2}-\d{4}$")
        assert df["social_security_number"].apply(lambda x: bool(ssn_pattern.match(str(x)))).all()
        email_pattern = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
        assert df["email_address"].apply(lambda x: bool(email_pattern.match(str(x)))).all()
        bp_pattern = re.compile(r"^\d{2,3}/\d{2,3}$")
        assert df["blood_pressure"].apply(lambda x: bool(bp_pattern.match(str(x)))).all()
        assert df["internal_doc_link"].str.startswith("https://hospital-intranet/").all()
        lot_pattern = re.compile(r"^LOT-[A-Z]{3}-\d{4}$")
        assert df["drug_lot_number"].apply(lambda x: bool(lot_pattern.match(str(x)))).all()
        uuid_pattern = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
        assert df["dicom_image_id"].apply(lambda x: bool(uuid_pattern.match(str(x)))).all()

        # Comparison syntax
        assert (df["short_diagnosis_code"].str.len().to_numpy() < df["full_diagnosis_description"].str.len().to_numpy()).all()
        assert (df["staff_initials"].str.len().to_numpy() < df["staff_last_name"].str.len().to_numpy()).all()
        assert (df["ward_code"].str.len() <= 5).all()

        # Amount comparisons
        assert (df["insurance_payout"] <= df["total_invoice_amount"]).all()
        assert (df["patient_copay"] + df["insurance_payout"] == df["total_invoice_amount"]).all()
        assert (df["anesthesia_dosage"] < df["max_safe_dosage"]).all()

        # Aggregation rules
        dept_sums = df.groupby("department_id")["staff_salaries"].sum()
        dept_budgets = df.groupby("department_id")["department_budget"].first().reindex(dept_sums.index)
        assert (dept_sums <= dept_budgets).all()

        ward_counts = df.groupby("ward_id").size()
        ward_beds = df.groupby("ward_id")["total_beds"].first().reindex(ward_counts.index)
        assert (ward_counts <= ward_beds).all()

        doctor_counts = df.groupby("doctor_id").size()
        assert (doctor_counts <= 20).all()

        # Monotonicity
        assert (df["recovery_score_day_1"] <= df["recovery_score_day_7"]).all()
        assert (df["pain_level_admission"] > df["pain_level_discharge"]).all()
        assert (df["medication_dosage_step_1"] < df["medication_dosage_step_2"]).all()

        # Outliers
        assert df["body_temperature_celsius"].between(35.0, 42.0).all()
        assert df["heart_rate_bpm"].between(30, 220).all()
        assert df["systolic_bp"].between(70, 200).all()


df = generate_dataset(NUM_ROWS)
validate_data(df)
df.to_csv("synthetic_dataset.csv", index=False)