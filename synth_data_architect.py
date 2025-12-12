import json
from datetime import datetime, timedelta
from typing import TypedDict, Annotated, List, Optional, Dict, Union
import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta
import traceback
import re  # Aggiungi re se non c'è, serve spesso per le regex

def extract_code_from_message(message):
    # 1. Recupera il contenuto (può essere stringa o lista di blocchi)
    content = message.content

    raw_text = ""

    # Se è una lista (nuovo formato GPT-5/Reasoning), cerca il blocco 'text'
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get('type') == 'text':
                raw_text = block.get('text', '')
                break
    # Se è già una stringa (formato classico)
    elif isinstance(content, str):
        raw_text = content

    # 2. Pulizia dei tag Markdown (```python ... ```)
    # Rimuove ```python all'inizio e ``` alla fine, e spazi bianchi extra
    clean_code = raw_text.replace("```python", "").replace("```", "").strip()

    return clean_code


class GraphState(TypedDict):
    """Stato del grafo per la generazione del dataset."""

    # Input iniziale
    user_instructions: str  # Schema e regole JSON/Testo
    output_path: str  # Dove salvare il CSV
    script_output_path: str  # <--- NUOVO: Dove salvare lo script Python

    # Stato interno
    generated_code: str  # Il codice Python corrente
    code_output: str  # Output (stdout) o Errore dell'esecuzione
    execution_success: bool  # Se il codice ha girato senza crash
    dataframe_obj: Optional[pd.DataFrame]  # Oggetto dataframe reale (in memoria)

    # Feedback e Controllo
    validation_error: str  # Errore logico (hallucination) o runtime
    iterations: int  # Contatore tentativi
    max_iterations: int  # Limite massimo (es. 4)

    error_history: List[str]


from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.prompts import ChatPromptTemplate
import traceback

# Inizializza il modello (o usa Gemini qui)
llm = ChatOpenAI(
    model="gpt-5.2",  # Oppure "gpt-5.1-codex-max" per task più complessi
    temperature=0,
    use_responses_api=True, # Fondamentale per i modelli della serie Codex/GPT-5
    reasoning_effort="high" # Opzionale: "low", "medium", "high" (solo per modelli reasoning)
)

llm_check = ChatOpenAI(model="gpt-5.2", temperature=0, reasoning_effort="high")

REFERENCE_CODE_CONTENT = """
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import re

# Configuration
NUM_ROWS = 1_000_000
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


# --- 1. GENERATION FUNCTIONS ---

def generate_reference_departments(n_depts=50):
    dept_ids = [f'D{i:03d}' for i in range(1, n_depts + 1)]
    dept_names = ['Executive Leadership'] + [f'Department_{i}' for i in range(2, n_depts + 1)]

    # Parent assignment: only index < current to avoid cycles (DAG)    parents = [None]
    for i in range(1, n_depts):
        parents.append(dept_ids[np.random.randint(0, i)])

    est_dates = [datetime(2000, 1, 1) + timedelta(days=np.random.randint(0, 5000)) for _ in range(n_depts)]

    return pd.DataFrame({
        'department_id': dept_ids,
        'department_name': dept_names,
        'parent_department_id': parents,
        'department_est_date': est_dates
    })


def generate_dataset(num_rows):
    print(f"--- Generation of {num_rows} rows... ---")

    # --- Referential ---
    df_depts = generate_reference_departments()

    # Job map
    job_map = [
        {'title': 'Software Engineer I', 'level': 'L1', 'rank': 1, 'family': 'Technology', 'base_min': 50000,
         'base_max': 60000},
        {'title': 'HR Specialist', 'level': 'L2', 'rank': 2, 'family': 'HR', 'base_min': 80000, 'base_max': 90000},
        # Gap L1-L2
        {'title': 'Software Engineer II', 'level': 'L2', 'rank': 2, 'family': 'Technology', 'base_min': 80000,
         'base_max': 90000},
        {'title': 'Senior Software Engineer', 'level': 'L3', 'rank': 3, 'family': 'Technology', 'base_min': 120000,
         'base_max': 130000},
        {'title': 'Lead Software Engineer', 'level': 'L4', 'rank': 4, 'family': 'Technology', 'base_min': 170000,
         'base_max': 180000},
        {'title': 'Marketing Manager', 'level': 'L4', 'rank': 4, 'family': 'Marketing', 'base_min': 170000,
         'base_max': 180000},
        {'title': 'VP of Operations', 'level': 'L6', 'rank': 6, 'family': 'Operations', 'base_min': 250000,
         'base_max': 260000},
        {'title': 'Intern', 'level': 'Intern', 'rank': 0, 'family': 'General', 'base_min': 30000, 'base_max': 35000},
    ]

    df = pd.DataFrame({'id': np.arange(1, num_rows + 1)})

    # Rule 9: Format Manager/Employee ID
    df['employee_id'] = [f'EMP-{i:06d}' for i in df['id']]

    # Departments assignment
    dept_indices = np.random.randint(0, len(df_depts), num_rows)
    df['department_id'] = df_depts.iloc[dept_indices]['department_id'].values
    df = df.merge(df_depts, on='department_id', how='left')

    # Job assignment
    job_indices = np.random.choice(len(job_map), num_rows, p=[0.2, 0.1, 0.2, 0.15, 0.05, 0.1, 0.05, 0.15])
    for key in ['job_title', 'job_level', 'level_rank', 'job_family', 'base_min', 'base_max']:
        if key == 'job_title':
            df[key] = [job_map[i]['title'] for i in job_indices]
        elif key == 'job_level':
            df[key] = [job_map[i]['level'] for i in job_indices]
        elif key == 'level_rank':
            df[key] = [job_map[i]['rank'] for i in job_indices]
        elif key == 'job_family':
            df[key] = [job_map[i]['family'] for i in job_indices]
        elif key == 'base_min':
            df[key] = [job_map[i]['base_min'] for i in job_indices]
        elif key == 'base_max':
            df[key] = [job_map[i]['base_max'] for i in job_indices]

    # Names and emails
    first_names = ['John', 'Jane', 'Michael', 'Emily', 'David', 'Sarah', 'James', 'Emma', 'Robert', 'Olivia']
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez']
    df['first_name'] = np.array(first_names)[np.random.randint(0, len(first_names), num_rows)]
    df['last_name'] = np.array(last_names)[np.random.randint(0, len(last_names), num_rows)]
    df['employee_name'] = df['first_name'] + ' ' + df['last_name']

    # Status and Date
    df['employment_status'] = np.random.choice(['Active', 'Terminated', 'Contractor'], num_rows, p=[0.85, 0.10, 0.05])
    start_date_base = datetime(2015, 1, 1)
    df['employment_start_date'] = [start_date_base + timedelta(days=int(x)) for x in
                                   np.random.randint(0, 3000, num_rows)]

    # Rule 2, 3, 4: Date Logic
    df['employment_end_date'] = pd.NaT
    term_mask = df['employment_status'] == 'Terminated'
    df.loc[term_mask, 'employment_end_date'] = df.loc[term_mask, 'employment_start_date'] + pd.to_timedelta(
        np.random.randint(1, 365, term_mask.sum()), unit='D')

    # Rule 8: Email Format
    df['email_base'] = (df['first_name'] + '.' + df['last_name']).str.lower()
    df['name_count'] = df.groupby('email_base').cumcount()
    df['email'] = df['email_base'] + df['name_count'].apply(lambda x: str(x) if x > 0 else '') + '@acme.corp'

    # Salaries e Bonus
    df['base_salary'] = np.random.uniform(df['base_min'], df['base_max'])
    df['bonus_percentage'] = np.random.uniform(0.05, 0.20, num_rows)

    # Rule 7: Bonus Cap
    mask_r7 = (df['level_rank'] < 5) & (df['department_id'] != 'D001')
    df.loc[mask_r7, 'bonus_percentage'] = df.loc[mask_r7, 'bonus_percentage'].clip(upper=0.3)

    # Rule 14: CFD Bonus (Bonus fisso per dept/level se rank <=3)
    bonus_fix_map = {}
    for dept in df['department_id'].unique():
        if dept == 'D001': continue
        for level in ['L1', 'L2', 'L3', 'Intern']:
            bonus_fix_map[(dept, level)] = round(random.uniform(0.05, 0.15), 4)

    mask_r14 = (df['department_name'] != 'Executive Leadership') & (df['level_rank'] <= 3)
    df['temp_key'] = list(zip(df['department_id'], df['job_level']))
    # Apply map only where needed
    fixed_bonuses = df.loc[mask_r14, 'temp_key'].map(bonus_fix_map)
    df.loc[mask_r14, 'bonus_percentage'] = fixed_bonuses

    # Rule 5: Contractor Bonus = 0 (Must be last)
    df.loc[df['employment_status'] == 'Contractor', 'bonus_percentage'] = 0.0

    # Rule 10: Total Comp
    df['total_compensation'] = df['base_salary'] * (1 + df['bonus_percentage'])

    # Manager e Buildings
    df['building_id'] = np.random.choice([f'B{i:02d}' for i in range(1, 11)], num_rows)
    df['city'] = np.random.choice(['NY', 'LDN', 'TOK'], num_rows)

    # --- CRITICAL FIXING FOR RULE 15 ---
    # An Intern cannot be a Manager. We select only Non-Interns.
    # This prevents the dependency chain (Intern -> Intern -> Manager).
    non_intern_ids = df.loc[df['job_level'] != 'Intern', 'employee_id']
    potential_managers = non_intern_ids.sample(n=min(50000, len(non_intern_ids))).values
    df['manager_id'] = np.random.choice(potential_managers, num_rows)

    # No self-manager 
    self_mg = df['manager_id'] == df['employee_id']
    # Shift manager id
    df.loc[self_mg, 'manager_id'] = np.roll(df.loc[self_mg, 'manager_id'], 1)

    # Rule 15: Intern Building
    # Ora che i manager sono stabili (non interns), possiamo assegnare l'edificio in sicurezza
    intern_mask = df['job_level'] == 'Intern'
    emp_build_map = dict(zip(df['employee_id'], df['building_id']))
    df.loc[intern_mask, 'building_id'] = df.loc[intern_mask, 'manager_id'].map(emp_build_map)

    # Budget (Rule 18)
    dept_sums = df.groupby('department_id')['total_compensation'].sum()
    df['department_budget'] = df['department_id'].map(dept_sums * 1.2)
    df['is_department_head'] = False  # Placeholder

    # Cleanup
    cols = ['id', 'employee_id', 'employee_name', 'email', 'job_title', 'job_level', 'job_family',
            'level_rank', 'department_id', 'department_name', 'parent_department_id',
            'department_est_date', 'building_id', 'city', 'manager_id', 'total_compensation',
            'base_salary', 'bonus_percentage', 'employment_status', 'employment_start_date',
            'employment_end_date', 'is_department_head', 'department_budget']

    return df[cols].copy()


# --- 2. COMPLETE VALIDATION (ALL THE RULES) ---

def validate_data(df):
    print("\n--- Execution validation (18 Rules) ---")

    # 1. Intra-Table Asymmetry (Dept != Parent)
    assert (df['department_id'] != df['parent_department_id']).all(), "R1 Violata: Dept ID uguale a Parent ID"
    print("R1 [Asymmetry]: OK")

    # 2. Temporal Order (End >= Start)
    mask_date = df['employment_end_date'].notna()
    assert (df.loc[mask_date, 'employment_end_date'] >= df.loc[
        mask_date, 'employment_start_date']).all(), "R2 Violata: Data Fine < Data Inizio"
    print("R2 [Temporal]: OK")

    # 3. Conditional Completeness (Terminated -> End Date)
    assert df.loc[df[
                      'employment_status'] == 'Terminated', 'employment_end_date'].notna().all(), "R3 Violata: Terminated con data fine nulla"
    print("R3 [Completeness Terminated]: OK")

    # 4. Conditional Completeness (Active -> No End Date)
    assert df.loc[df[
                      'employment_status'] == 'Active', 'employment_end_date'].isna().all(), "R4 Violata: Active con data fine valorizzata"
    print("R4 [Completeness Active]: OK")

    # 5. Value Association (Contractor -> No Bonus)
    assert (df.loc[df[
                       'employment_status'] == 'Contractor', 'bonus_percentage'] == 0).all(), "R5 Violata: Contractor ha bonus > 0"
    print("R5 [Contractor Bonus]: OK")

    # 6. Value Association (Job Titles -> Technology)
    tech_titles = ['Software Engineer I', 'Software Engineer II', 'Senior Software Engineer', 'Lead Software Engineer']
    mask_tech = df['job_title'].isin(tech_titles)
    assert (df.loc[mask_tech, 'job_family'] == 'Technology').all(), "R6 Violata: Job Family errata per Tech Titles"
    print("R6 [Job Family]: OK")

    # 7. Value Association (Bonus Cap 0.3)
    # Condizione: Rank < 5 AND Dept != D001 AND Start Date > 10 years ago (qui tutte lo sono)
    mask_r7 = (df['level_rank'] < 5) & (df['department_id'] != 'D001')
    assert (df.loc[mask_r7, 'bonus_percentage'] <= 0.30001).all(), "R7 Violata: Bonus > 0.3 per rank basso"
    print("R7 [Bonus Cap]: OK")

    # 8. Format Checking (Email)
    # Pattern: name.surname[opt_number]@acme.corp
    email_pattern = r'^[a-z]+\.[a-z]+(\d+)?@acme\.corp$'
    assert df['email'].str.match(email_pattern).all(), "R8 Violata: Formato Email non valido"
    print("R8 [Email Format]: OK")

    # 9. Domain-Specific Format (Manager ID)
    assert df['manager_id'].str.match(r'^EMP-\d{6}$').all(), "R9 Violata: Manager ID format invalid"
    print("R9 [Manager ID Format]: OK")

    # 10. Amount Comparison (Total >= Base)
    assert (df['total_compensation'] >= df['base_salary']).all(), "R10 Violata: Total Comp < Base"
    print("R10 [Amount Comparison]: OK")

    # 11. Functional Dependency (Dept, Level -> Rank)
    fds = df.groupby(['department_id', 'job_level'])['level_rank'].nunique()
    assert (fds == 1).all(), "R11 Violata: FD Dept+Level -> Rank fallita"
    print("R11 [FD Checking]: OK")

    # 12. Monotonicity Rule
    # Compensi non decrescenti per Level Rank (all'interno di Dept e Family)
    df_sorted = df.sort_values(['department_id', 'job_family', 'level_rank', 'total_compensation'])

    # Shift to compare the previous row
    prev_dept = df_sorted['department_id'].shift(1)
    prev_fam = df_sorted['job_family'].shift(1)
    prev_rank = df_sorted['level_rank'].shift(1)
    prev_comp = df_sorted['total_compensation'].shift(1)

    check_mask = (df_sorted['department_id'] == prev_dept) & \
                 (df_sorted['job_family'] == prev_fam) & \
                 (df_sorted['level_rank'] > prev_rank)

    violations = df_sorted[check_mask & (df_sorted['total_compensation'] < prev_comp)]
    assert violations.empty, "R12 Violata: Monotonicità del compenso non rispettata"
    print("R12 [Monotonicity]: OK")

    # 13. Outlier Detection (Skipped as soft rule)

    # 14. CFD (Bonus Equality)
    mask_r14 = (df['department_name'] != 'Executive Leadership') & (df['level_rank'] <= 3)
    target_df = df[mask_r14]
    # Filtriamo Contractor per evitare falsi positivi (loro hanno bonus 0 sempre)
    target_df_clean = target_df[target_df['employment_status'] != 'Contractor']
    if not target_df_clean.empty:
        bonus_counts_clean = target_df_clean.groupby(['department_id', 'job_level'])['bonus_percentage'].nunique()
        assert (bonus_counts_clean <= 1).all(), "R14 Violata: Bonus non uniforme nel gruppo target"
    print("R14 [CFD Bonus]: OK")

    # 15. Intra-Table Equality (Intern Building)
    # Check su campione significativo
    interns = df[df['job_level'] == 'Intern']
    if not interns.empty:
        mgr_map = df.set_index('employee_id')['building_id']
        expected_bldgs = interns['manager_id'].map(mgr_map)
        # Verifica se ci sono manager mancanti (dovrebbe essere impossibile per costruzione) o mismatch
        assert (interns['building_id'] == expected_bldgs).all(), "R15 Violata: Intern Building != Manager Building"
    print("R15 [Intern Equality]: OK")

    # 16. Unique Key (Active -> Employee ID)
    active_ids = df.loc[df['employment_status'] == 'Active', 'employee_id']
    assert active_ids.is_unique, "R16 Violata: Employee ID duplicati per Active"
    print("R16 [Unique Key]: OK")

    # 17. Advanced Duplicate Detection (Warning Only)

    # 18. Aggregated Amount (Budget)
    dept_agg = df.groupby('department_id').agg({'total_compensation': 'sum', 'department_budget': 'first'})
    assert (dept_agg['department_budget'] >= dept_agg[
        'total_compensation'] - 0.01).all(), "R18 Violata: Budget < Sum(Comp)"
    print("R18 [Aggregated Budget]: OK")

    print("\nSUCCESS: TUTTE LE REGOLE SONO STATE RISPETTATE.")


df = generate_dataset(NUM_ROWS)
validate_data(df)
df.to_csv("synthetic_dataset_validated.csv", index=False)
"""

SYNTH_DATA_ARCHITECT_PROMPT = """You are **SynthData Architect**, an AI assistant specialized in creating high-quality synthetic datasets for testing, ML training, and stress testing.

### YOUR GOAL
Your task is to write **complete, efficient, and ready-to-use Python code** that generates CSV files containing millions of rows based on a provided schema and specific natural language rules.

### WORK PROCESS
Always follow these logical steps internally:

1.  **Schema Analysis:** Understand the column names, data types, and required structure.
2.  **Rule Integration (User + KB):**
    * Strictly apply the rules provided by the user.
    * Consult your **Reference Architecture** (below) for coding standards, vectorization techniques, and validation structure.
    * *Priority:* User-specific rules always override general reference patterns.
3.  **Optimization Strategy:**
    * Use vectorization libraries like `pandas` and `numpy`.
    * Use `faker` for realistic qualitative data.
    * **Avoid slow loops** (iterating rows). Use vectorized operations (`np.where`, `map`, `apply` on logic masks).
4.  **Validation Implementation:** Your script MUST include a final `validate_data(df)` function that prints "SUCCESS" or raises assertions/errors if rules are violated.

### REFERENCE ARCHITECTURE (GOLD STANDARD)
Use the following code structure and logic complexity as your guide. Note how it handles generation, optimization, and rigorous validation:
'''python
{reference_code}
'''

### PYTHON CODE REQUIREMENTS
* **Libraries:** Use `pandas`, `numpy`, `faker`, `random`.
* **Reproducibility:** Always set a `random seed`.
* **Output:** The code must save the result to a `.csv` file (e.g., 'dataset.csv').
* **Variable Name:** The final dataframe must be named `df`.

### STRICT OUTPUT FORMAT
You must output **ONLY** the Python code inside a markdown code block.
* **DO NOT** write any conversational text.
* **DO NOT** provide explanations.
* **DO NOT** ask follow-up questions.
* Your response must start directly with ```python and end immediately after the code block closes.
"""



def code_generator_node(state: GraphState):
    print(f"--- CODE GENERATION (Attempt {state['iterations'] + 1}) ---")

    # Recupera lo storico attuale (o lista vuota se None)
    current_history = state.get("error_history", [])

    # Se c'è un errore dal tentativo precedente, lo aggiungiamo allo storico
    if state.get("validation_error"):
        new_error_entry = (
            f"Attempt {state['iterations']}:\n"
            f"Error: {state['validation_error']}\n"
            f"Code snippet responsible (context): See previous code execution."
        )
        current_history.append(new_error_entry)

    user_msg_content = f"**USER INSTRUCTIONS & RULES:**\n{state['user_instructions']}\n\nSuggestion: Do not use 'if __name__ == __main__' directly."

    # Se abbiamo uno storico di errori, lo iniettiamo nel prompt
    if current_history:
        history_text = "\n\n".join(current_history)
        user_msg_content += (
            f"\n\n!!! HISTORY OF PREVIOUS FAILURES !!!\n"
            f"You have already tried to generate this code {len(current_history)} times. "
            f"Here is the log of past errors you MUST avoid repeating:\n"
            f"==================================================\n"
            f"{history_text}\n"
            f"==================================================\n"
            f"CRITICAL INSTRUCTION: Analyze the entire history above. "
            f"Do not revert to old code that caused these errors. "
            f"Implement a NEW solution that fixes the latest error without re-introducing previous ones."
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYNTH_DATA_ARCHITECT_PROMPT),
        ("user", "{user_msg}")
    ])

    chain = prompt | llm

    response = chain.invoke({
        "reference_code": REFERENCE_CODE_CONTENT,
        "user_msg": user_msg_content
    })

    # --- Logica di estrazione codice (invariata) ---
    content_data = response.content
    code_text = ""
    if isinstance(content_data, list):
        for block in content_data:
            if isinstance(block, dict) and block.get('type') == 'text':
                code_text = block['text']
                break
    else:
        code_text = str(content_data)

    clean_code = code_text.replace("```python", "").replace("```", "").strip()
    # -----------------------------------------------

    print(f"Code generated (Length: {len(clean_code)} chars)")

    return {
        "generated_code": clean_code,
        "iterations": state["iterations"] + 1,
        "validation_error": None,  # Resettiamo l'errore corrente perché stiamo provando una fix
        "error_history": current_history  # Passiamo lo storico aggiornato al prossimo step
    }


def code_executor_node(state: GraphState):
    print("--- CODE EXECUTION ---")
    code = state["generated_code"]

    execution_context = {
        "pd": pd,
        "np": np,
        "random": random,
        "Faker": Faker,
        "fake": Faker('it_IT'),
        "datetime": datetime,
        "timedelta": timedelta,
        "re": re,
        "__builtins__": __builtins__
    }

    try:
        exec(code, execution_context)

        if "df" not in execution_context:
            raise ValueError("'df' was not generated.")

        df = execution_context["df"]

        # Validazione e output
        sample = df.head(10).to_markdown()

        return {
            "execution_success": True,
            "code_output": "Execution completed with success.",
            "dataset_sample": sample,
            "dataframe_obj": df,
            "validation_error": None
        }

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"--- RUNTIME ERROR: {error_msg} ---")
        return {
            "execution_success": False,
            "code_output": error_msg,
            "validation_error": error_msg
        }

def hallucination_check_node(state: GraphState):
    print("--- VERIFY LOGIC (HALLUCINATION CHECK) ---")

    rules = state["user_instructions"]
    code = state["generated_code"]

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a Quality Assurance Engineer. Analyze the assertions in the code and verify that all user-specified rules are correctly checked by the assertions."),
        ("user",
         "USER RULES:\n{rules}\n\nCODE WITH ASSERTS:\n{code}\n\nReply ONLY with 'OK' if it is perfect, or describe the logical error if it fails.")
    ])

    chain = prompt | llm_check
    response = chain.invoke({
        "rules": rules,
        "code": code
    })
    feedback = response.content

    if feedback.strip().upper() == "OK":
        return {"validation_error": None}
    else:
        print(f"--- VALIDATION FAULT: {feedback} ---")
        return {"validation_error": feedback}  # Questo feedback torna al generatore


# --- NODO 4: SALVATAGGIO FILE INTELLIGENTE (CSV + PYTHON + WARNING) ---
def file_saver_node(state: GraphState):
    print("--- SAVING OUTPUTS ---")

    csv_path = state["output_path"]
    script_path = state["script_output_path"]

    # Costruiamo il path per il warning report (es. dataset_WARNING.txt)
    base_name = csv_path.rsplit('.', 1)[0]
    warning_path = f"{base_name}_WARNING_REPORT.txt"

    # 1. Salvataggio Script Python (Sempre utile per il debug)
    if state["generated_code"]:
        try:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(state["generated_code"])
            print(f"✅ Python Script saved to: {script_path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not save python script: {e}")

    # 2. Salvataggio CSV (Solo se il dataframe esiste)
    df = state["dataframe_obj"]
    if df is not None:
        try:
            df.to_csv(csv_path, index=False)
            print(f"✅ CSV saved to: {csv_path}")
        except Exception as e:
            print(f"❌ Error saving CSV: {e}")
    else:
        print("⚠️ No DataFrame generated to save (Execution crashed).")

    # 3. Salvataggio Warning Report (Solo se c'è un errore attivo)
    # Verifichiamo se siamo qui perché è fallita la validazione o l'esecuzione
    error_msg = state.get("validation_error")

    # Se non c'è errore di validazione, magari c'è stato un crash di esecuzione
    if not error_msg and not state.get("execution_success"):
        error_msg = state.get("code_output")  # Prende lo stacktrace dell'errore

    if error_msg:
        try:
            report_content = (
                f"⚠️ SYNTHETIC DATA GENERATION FAILED (PARTIALLY) ⚠️\n"
                f"==================================================\n"
                f"Timestamp: {datetime.now()}\n"
                f"Attempts made: {state['iterations']}\n\n"
                f"CRITICAL ERROR DETAILS:\n"
                f"-----------------------\n"
                f"{error_msg}\n\n"
                f"STATUS:\n"
                f"- Python Script: Saved ({script_path})\n"
                f"- Dataset CSV: {'Saved (Potential logical errors)' if df is not None else 'Not Saved (Code crash)'}\n"
            )

            with open(warning_path, "w", encoding="utf-8") as f:
                f.write(report_content)

            print(f"🚨 WARNING REPORT saved to: {warning_path}")
            return {"code_output": f"Process completed with WARNINGS. See {warning_path}"}

        except Exception as e:
            print(f"❌ Error saving warning report: {e}")

    return {"code_output": "Process completed successfully."}


from langgraph.graph import StateGraph, END

def route_after_execution(state: GraphState):
    if not state["execution_success"]:
        if state["iterations"] >= state["max_iterations"]:
            return "failed"
        return "retry_coding"
    return "check_hallucination"

def route_after_check(state: GraphState):
    if state["validation_error"]:
        if state["iterations"] >= state["max_iterations"]:
            return "failed"
        return "retry_coding"
    return "save_file"


def build_synt_data_agent():
    workflow = StateGraph(GraphState)

    workflow.add_node("generator", code_generator_node)
    workflow.add_node("executor", code_executor_node)
    workflow.add_node("validator", hallucination_check_node)

    # NOTA: Usiamo file_saver_node sia per il successo che per il fallimento
    workflow.add_node("saver", file_saver_node)

    # Il nodo fail_end diventa ridondante se vogliamo sempre salvare,
    # ma possiamo tenerlo per logica interna o rimuoverlo.
    # Qui lo rimuoviamo dal flusso "failed" delle conditional edges.

    workflow.set_entry_point("generator")
    workflow.add_edge("generator", "executor")

    # Routing dopo Esecuzione
    workflow.add_conditional_edges(
        "executor",
        route_after_execution,
        {
            "retry_coding": "generator",
            "check_hallucination": "validator",
            "failed": "saver"  # <--- CAMBIAMENTO: Se fallisce max retries, vai a salvare
        }
    )

    # Routing dopo Validazione
    workflow.add_conditional_edges(
        "validator",
        route_after_check,
        {
            "retry_coding": "generator",
            "save_file": "saver",  # Successo
            "failed": "saver"  # <--- CAMBIAMENTO: Se fallisce max retries, vai a salvare comunque
        }
    )

    workflow.add_edge("saver", END)

    app = workflow.compile()
    return app


# --- PUBLIC PROXY FUNCTION ---
def generate_synthetic_dataset(
        num_rows: int,
        schema: Union[List[str], str],
        rules: List[Dict],
        output_path: str = "synthetic_dataset.csv",
        script_output_path: Optional[str] = None,  # <--- NUOVO PARAMETRO
        max_retries: int = 3
) -> Optional[pd.DataFrame]:
    """
    Proxy function to generate synthetic datasets, hiding LangGraph complexity.

    Args:
        ...
        script_output_path (str, optional): Path where to save the .py generation script.
                                            If None, defaults to output_path with .py extension.
    """

    # Logica per determinare il nome del file script se non fornito
    if script_output_path is None:
        # Se output_path è "data.csv", script diventa "data_generation.py"
        base_name = output_path.rsplit('.', 1)[0]
        script_output_path = f"{base_name}_gen.py"

    # 1. Build Structured User Prompt
    if isinstance(schema, list):
        schema_str = ", ".join(schema)
    else:
        schema_str = schema

    rules_str = json.dumps(rules, indent=2)

    user_prompt = (
        f"Generate a dataset containing exactly {num_rows} rows.\n\n"
        f"### DATASET SCHEMA:\n"
        f"{schema_str}\n\n"
        f"### STRICT RULES (All the rules MUST BE respected in the generation!):\n"
        f"{rules_str}"
    )

    # 2. Initialize Graph State
    initial_state = {
        "user_instructions": user_prompt,
        "output_path": output_path,
        "script_output_path": script_output_path,  # <--- Inseriamo il path nello stato
        "iterations": 0,
        "max_iterations": max_retries,
        "validation_error": None,
        "execution_success": False,
        "generated_code": "",
        "code_output": "",
        "dataframe_obj": None,
        "error_history": [],
    }

    print(f"🚀 Starting Dataset generation ({num_rows} rows)...")
    print(f"ℹ️ Rules: {len(rules)} constraints defined")

    # 3. Invoke Graph
    try:
        final_state = build_synt_data_agent().invoke(initial_state)
    except Exception as e:
        print(f"❌ Critical error during graph execution: {e}")
        traceback.print_exc()
        return None

    # 4. Handle Result (Logica aggiornata)
    df = final_state.get("dataframe_obj")
    validation_err = final_state.get("validation_error")
    execution_success = final_state.get("execution_success")

    # CASO A: Successo Totale
    if df is not None and not validation_err and execution_success:
        print(f"\n✅ GENERATION COMPLETED SUCCESSFULLY!")
        print(f"💾 CSV saved to: {output_path}")
        print(f"🐍 Script saved to: {script_output_path}")
        return df

    # CASO B: Fallimento Parziale (CSV generato ma regole violate)
    elif df is not None:
        print(f"\n⚠️ GENERATION COMPLETED WITH WARNINGS.")
        print(f"The dataset was generated and saved, BUT logic validation failed.")
        print(f"💾 CSV saved to: {output_path} (Check carefully!)")
        print(f"🐍 Script saved to: {script_output_path}")
        print(f"🚨 Read the warning report for details.")
        return df  # Restituiamo comunque il DF per permettere all'utente di ispezionarlo

    # CASO C: Fallimento Totale (Crash codice, niente CSV)
    else:
        print(f"\n❌ GENERATION FAILED COMPLETELY (No CSV created).")
        print(f"🐍 Script saved to: {script_output_path} (for debugging)")
        print(f"Execution Error: {final_state.get('code_output')}")
        return None