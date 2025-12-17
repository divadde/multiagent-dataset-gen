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

from prompts import SYNTH_DATA_ARCHITECT_PROMPT_NEW


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
    model="gpt-5.1-codex-max",  # Oppure "gpt-5.1-codex-max" per task più complessi
    temperature=0,
    use_responses_api=True, # Fondamentale per i modelli della serie Codex/GPT-5
    reasoning_effort="high" # Opzionale: "low", "medium", "high" (solo per modelli reasoning)
)

llm_check = ChatOpenAI(model="gpt-5.2", temperature=0, reasoning_effort="high")

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
        ("system", "{system_prompt}"), #"{reference_code}"
        ("user", "{user_msg}")
    ])

    # Debugging
    #print(prompt.invoke({"user_msg": user_msg_content, "reference_code": REFERENCE_CODE_CONTENT}))

    chain = prompt | llm

    response = chain.invoke({
        "system_prompt": SYNTH_DATA_ARCHITECT_PROMPT_NEW,
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
         "You are a Quality Assurance Engineer. Analyze the assertions in the code and verify that all user-specified rules are correctly checked by the assertions (No need to verify the assertion of the number of rows)."),
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
        max_retries: int = 4
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