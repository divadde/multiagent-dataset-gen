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


# --- NODO MASTER (Stato) ---
def master_node(state: GraphState):
    """
    Nodo 'dummy' che segna l'inizio del processo.
    In LangGraph i nodi devono restituire un dict (aggiornamento stato).
    """
    print("\n🚀 [MASTER] Pipeline Started. Preparing task distribution...")
    return {}  # Nessun aggiornamento di stato, prosegue al conditional edge


# --- LOGICA DI DISTRIBUZIONE (Conditional Edge) ---
def distribute_tasks(state: GraphState):
    """
    Questa funzione viene chiamata dall'edge condizionale.
    Esegue l'algoritmo di planning e restituisce gli oggetti Send
    per avviare i worker in parallelo.
    """
    print("   📊 [MASTER] Analyzing dependencies & bin-packing rules...")

    rules = state["original_rules"]

    # 1. Chiama l'algoritmo di allocazione (Bin Packing)
    # Imposta max_rules a 8 (o meno) per evitare sovraccarico cognitivo dell'LLM
    batches = plan_generation_tasks(rules, max_rules=8)

    print(f"   📋 [MASTER] Created {len(batches)} parallel tasks.")
    for b in batches:
        print(f"      -> Batch {b['batch_id']}: {b['total_rules']} rules, Columns: {len(b['all_columns'])}")

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