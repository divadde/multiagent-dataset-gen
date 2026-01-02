from typing import TypedDict, Annotated, List, Optional, Dict, Union
import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta
import re  # Aggiungi re se non c'è, serve spesso per le regex
from utils.prompts import SYNTH_DATA_ARCHITECT_PROMPT
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.prompts import ChatPromptTemplate
import traceback
from langgraph.graph import StateGraph, END
from utils.utils import extract_code_from_message


class GraphState(TypedDict):
    """Stato interno per il ciclo di generazione/validazione del singolo worker."""
    # Input
    batch_id: int
    subset_rules: List[Dict]
    subset_columns: List[str]
    num_rows: int

    # Processo
    generated_code: str          # La funzione Python generata
    code_output: str             # Risultato esecuzione o stacktrace errore
    execution_success: bool      # Flag successo esecuzione
    dataframe_obj: Optional[pd.DataFrame] # DF parziale per validazione

    # Feedback loop
    validation_error: Optional[str]
    iterations: int
    max_retries: int
    error_history: List[str]

# Inizializza il modello (o usa Gemini qui)
llm = ChatOpenAI(
    model="gpt-5.1-codex-max",  # Oppure "gpt-5.1-codex-max" per task più complessi
    temperature=0,
    use_responses_api=True, # Fondamentale per i modelli della serie Codex/GPT-5
    reasoning_effort="high" # Opzionale: "low", "medium", "high" (solo per modelli reasoning)
)

llm_check = ChatOpenAI(model="gpt-5.2", temperature=0, reasoning_effort="high")

def code_generator_node(state: GraphState):
    print(f"   ⚙️ [Worker {state['batch_id']}] Attempt {state['iterations'] + 1} generating code...")

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

    # Costruzione Prompt
    rules_str = str(state["subset_rules"])
    cols_str = ", ".join(state["subset_columns"])
    num_rows = state["num_rows"]

    user_msg_content = f"""### CONFIGURATION
- Function Name: `generate_dataset(num_rows)`
- Target Columns: {cols_str}
- Rows to generate: {num_rows}

### STRICT BUSINESS RULES (Apply ALL of them):
{rules_str}"""

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
        ("system", "{system_prompt}"), #"{reference_code}"
        ("user", "{user_msg}")
    ])

    # Debugging
    #print(prompt.invoke({"user_msg": user_msg_content, "reference_code": REFERENCE_CODE_CONTENT}))

    chain = prompt | llm

    response = chain.invoke({
        "system_prompt": SYNTH_DATA_ARCHITECT_PROMPT,
        #"reference_code": REFERENCE_CODE_CONTENT,
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

    clean_code = extract_code_from_message(code_text)
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

        # 2. Verifica che la funzione esista
        if "generate_dataset" not in execution_context:
            raise ValueError(f"Function \"generate_dataset\" not found in generated code.")

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
    print(f"   🕵️ [Worker {state['batch_id']}] Validating logic...")

    rules = state["subset_rules"]
    code = state["generated_code"]

    # Usiamo un LLM Checker simile al file originale
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a QA Engineer. Verify that the Python code implements the following constraints."),
        ("user",
         "RULES:\n{rules}\n\nCODE:\n{code}\n\nDoes the code implement checks or logic for these rules? Reply 'OK' or describe the missing logic.")
    ])

    chain = prompt | llm_check
    response = chain.invoke({"rules": str(rules), "code": code})
    feedback = response.content.strip()

    if feedback.upper() == "OK" or "OK" in feedback.upper()[:5]:
        return {"validation_error": None}
    else:
        print(f"   ⚠️ [Worker {state['batch_id']}] Logic Check Failed: {feedback}")
        return {"validation_error": feedback}


# --- NODO 4: SALVATAGGIO FILE INTELLIGENTE (CSV + PYTHON + WARNING) ---
def file_saver_node(state: GraphState):
    """Salva gli artefatti del worker con il batch_id nel nome."""
    batch_id = state["batch_id"]
    print(f"   💾 [Worker {batch_id}] Saving intermediate artifacts...")

    # Costruiamo nomi file univoci per il batch
    base_name = f"batch_{batch_id}"
    script_path = f"{base_name}_code.py"
    csv_path = f"{base_name}_data.csv"
    warning_path = f"{base_name}_WARNING_REPORT.txt"

    # 1. Salvataggio Script
    if state["generated_code"]:
        try:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(state["generated_code"])
        except Exception as e:
            print(f"   ⚠️ [Worker {batch_id}] Could not save script: {e}")

    # 2. Salvataggio CSV (se disponibile)
    if state["dataframe_obj"] is not None:
        try:
            state["dataframe_obj"].to_csv(csv_path, index=False)
        except Exception as e:
            print(f"   ❌ [Worker {batch_id}] Error saving CSV: {e}")

    # 3. Salvataggio Warning Report (se c'è errore)
    error_msg = state.get("validation_error")
    # Se validation_error è None ma execution_success è False, prendiamo l'errore runtime
    if not error_msg and not state.get("execution_success"):
        error_msg = state.get("code_output")

    if error_msg:
        try:
            report_content = (
                f"⚠️ BATCH {batch_id} GENERATION FAILED ⚠️\n"
                f"Timestamp: {datetime.now()}\n"
                f"Attempts made: {state['iterations']}\n"
                f"Columns: {state['subset_columns']}\n\n"
                f"ERROR DETAILS:\n{error_msg}\n"
            )
            with open(warning_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            print(f"   🚨 [Worker {batch_id}] Warning report saved: {warning_path}")
        except Exception as e:
            print(f"   ❌ [Worker {batch_id}] Error saving warning: {e}")

    # Ritorniamo un output informativo, ma non cambiamo lo stato critico
    return {"code_output": "Artifacts saved."}

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


def worker_generator_node(state):
    """
    Nodo Entry Point chiamato dal Master tramite Send().
    """
    batch_id = state["batch_id"]

    initial_internal_state = {
        "batch_id": batch_id,
        "subset_rules": state["subset_rules"],
        "subset_columns": state["subset_columns"],
        "num_rows": state["num_rows"],
        "generated_code": "",
        "code_output": "",
        "execution_success": False,
        "dataframe_obj": None,
        "validation_error": None,
        "iterations": 0,
        "max_retries": 3,
        "error_history": []
    }

    # Esegue il ciclo completo (Gen->Exec->Valid->Save)
    final_internal_state = build_synt_data_agent().invoke(initial_internal_state)

    # Recupera il codice finale (anche se parzialmente corretto)
    final_code = final_internal_state["generated_code"]

    return {"generated_code_snippets": [final_code]}