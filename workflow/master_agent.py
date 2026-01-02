from typing import TypedDict, List, Annotated, Dict
import operator
from langgraph.constants import Send
from utils.analyze_rule_dependencies import plan_generation_tasks


# --- STATO GLOBALE (Condiviso) ---
class GraphState(TypedDict):
    user_instructions: str  # Schema e info generali
    original_rules: List[Dict]  # Tutte le regole
    num_rows: int

    # Accumulatore: raccoglie i pezzi di codice generati dai worker
    generated_code_snippets: Annotated[List[str], operator.add]

    final_combined_code: str  # Risultato finale (dopo il merge)


# --- NODO MASTER ---
def master_planner_node(state: GraphState):
    print("\n🚀 [MASTER] Analyzing dependencies & planning tasks...")

    rules = state["original_rules"]

    # 1. Chiama l'algoritmo di allocazione
    batches = plan_generation_tasks(rules, max_rules=8)

    print(f"📋 [MASTER] Created {len(batches)} parallel tasks.")

    # 2. Crea gli oggetti Send per il mapping parallelo
    # Manda ogni batch al nodo "worker_node"
    return [
        Send("worker_node", {
            "batch_id": batch["batch_id"],
            "subset_rules": batch["all_rules"],
            "subset_columns": batch["all_columns"],
            "num_rows": state["num_rows"]
        })
        for batch in batches
    ]


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
    final_internal_state = worker_app.invoke(initial_internal_state)

    # Recupera il codice finale (anche se parzialmente corretto)
    final_code = final_internal_state["generated_code"]

    return {"generated_code_snippets": [final_code]}