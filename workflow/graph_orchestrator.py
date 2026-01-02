import concurrent

from langgraph.graph import StateGraph, START, END
from workflow.master_agent import GraphState, master_node, distribute_tasks
from workflow.synth_data_architect import worker_generator_node
import json
import re
import pandas as pd
import numpy as np


# --- NODO MERGER (Rinomina, Esecuzione Parallela, Assemblaggio) ---
def code_merger_node(state: GraphState):
    """
    1. Riceve snippet (tutti con funzione 'generate_dataset').
    2. Li RINOMINA dinamicamente in 'generate_batch_0', 'generate_batch_1', etc.
    3. ESEGUE le funzioni rinominate IN PARALLELO.
    4. Unisce i risultati e produce lo script finale.
    """
    print("\n🔗 [MERGER] Renaming functions and executing in PARALLEL...")

    snippets = state["generated_code_snippets"]
    num_rows = state["num_rows"]

    function_names = []
    functions_to_run = {}

    # Contesto di esecuzione con le librerie necessarie
    exec_context = {
        "pd": pd,
        "np": np,
        "random": __import__("random"),
        "Faker": __import__("faker").Faker,
        "datetime": __import__("datetime"),
        "timedelta": __import__("datetime").timedelta
    }

    # --- 1. COSTRUZIONE SCRIPT E DEFINIZIONE DINAMICA ---

    final_script = (
        "import pandas as pd\n"
        "import numpy as np\n"
        "from faker import Faker\n"
        "import random\n"
        "from datetime import datetime, timedelta\n\n"
        "# --- GENERATED PARTIAL FUNCTIONS ---\n\n"
    )

    for i, snippet in enumerate(snippets):
        unique_gen_name = f"generate_batch_{i}"
        unique_val_name = f"validate_dataset_{i}"  # <--- NUOVO NOME UNIVOCO PER VALIDATOR

        # --- FIX IMPORTANTE: SOSTITUZIONE GLOBALE ---
        # Sostituiamo 'generate_dataset' ovunque appaia come parola intera (\b)
        modified_snippet = re.sub(
            r"\bgenerate_dataset\b",
            unique_gen_name,
            snippet
        )

        # --- NUOVA FIX: SOSTITUZIONE VALIDATE_DATA ---
        # Sostituiamo 'validate_data' ovunque (definizione e chiamata) con 'validate_data_i'
        # Questo previene conflitti se i worker definiscono funzioni di validazione identiche
        modified_snippet = re.sub(
            r"\bvalidate_dataset\b",
            unique_val_name,
            modified_snippet
        )

        # Aggiungiamo lo snippet MODIFICATO allo script finale
        final_script += modified_snippet + "\n\n"

        # Eseguiamo exec() sul codice MODIFICATO
        try:
            exec(modified_snippet, exec_context)

            # Verifichiamo che la funzione di generazione esista nel contesto
            if unique_gen_name in exec_context:
                function_names.append(unique_gen_name)
                functions_to_run[unique_gen_name] = exec_context[unique_gen_name]
            else:
                # Fallback check
                found = False
                for key in exec_context:
                    if key == unique_gen_name:
                        found = True
                        break
                if not found:
                    print(f"❌ [MERGER] Function {unique_gen_name} not found after exec.")

        except Exception as e:
            print(f"❌ [MERGER] Error defining functions for batch {i}: {e}")

    function_names.sort()

    # --- 2. ESECUZIONE PARALLELA (ThreadPool) ---

    partial_dfs = []
    print(f"   ⚡ Launching {len(functions_to_run)} generation tasks in parallel...")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Mappa: Future -> Nome Funzione
        future_to_name = {
            executor.submit(func, num_rows): name
            for name, func in functions_to_run.items()
        }

        for future in concurrent.futures.as_completed(future_to_name):
            name = future_to_name[future]
            try:
                df_partial = future.result()
                if isinstance(df_partial, pd.DataFrame):
                    partial_dfs.append(df_partial)
                    print(f"      ✅ {name} completed ({len(df_partial.columns)} cols)")
                else:
                    print(f"      ⚠️ {name} returned invalid type: {type(df_partial)}")
            except Exception as exc:
                print(f"      ❌ {name} generated an exception: {exc}")

    # --- 3. CONCATENAZIONE E DATAFRAME FINALE ---

    final_df_obj = None
    if partial_dfs:
        print(f"   🔗 Concatenating {len(partial_dfs)} partial DataFrames...")
        try:
            # Concatenazione orizzontale
            final_df_obj = pd.concat(partial_dfs, axis=1)

            # Rimuoviamo colonne duplicate (es. chiavi primarie duplicate dai vari worker)
            final_df_obj = final_df_obj.loc[:, ~final_df_obj.columns.duplicated()]

            print(f"   ✅ Final Dataset ready: {final_df_obj.shape}")
        except Exception as e:
            print(f"   ❌ Error during concatenation: {e}")

    # --- 4. COMPLETAMENTO SCRIPT FINALE (Main Block) ---

    final_script += "# --- MAIN ORCHESTRATOR ---\n"
    final_script += "def generate_full_dataset(num_rows):\n"
    final_script += "    print(f'Starting generation for {num_rows} rows...')\n"
    final_script += "    partial_dfs = []\n\n"

    for func in function_names:
        final_script += f"    print('Running {func}...')\n"
        final_script += f"    partial_dfs.append({func}(num_rows))\n"

    final_script += "\n    # Merge orizzontale\n"
    final_script += "    print('Merging all columns...')\n"
    final_script += "    final_df = pd.concat(partial_dfs, axis=1)\n"
    final_script += "    # Remove duplicate columns\n"
    final_script += "    final_df = final_df.loc[:, ~final_df.columns.duplicated()]\n"
    final_script += "    return final_df\n\n"

    final_script += "if __name__ == '__main__':\n"
    final_script += f"    NUM_ROWS = {num_rows}\n"
    final_script += "    df = generate_full_dataset(NUM_ROWS)\n"
    final_script += "    output_file = 'synthetic_dataset_parallel.csv'\n"
    final_script += "    df.to_csv(output_file, index=False)\n"
    final_script += "    print(f'✅ Success! Dataset saved to {output_file}')\n"

    return {
        "final_combined_code": final_script,
        "dataframe_obj": final_df_obj
    }


# --- COSTRUZIONE DEL GRAFO ---
def build_parallel_graph():
    workflow = StateGraph(GraphState)

    # 1. Aggiungi il nodo Master (che ora è un nodo standard)
    workflow.add_node("master", master_node)

    # 2. Aggiungi il nodo Worker
    workflow.add_node("worker_node", worker_generator_node)

    # 3. Aggiungi il nodo Merger
    workflow.add_node("merger", code_merger_node)

    # Entry point -> Master
    workflow.add_edge(START, "master")

    # Map-Reduce:
    # L'uscita del nodo "master" attiva la funzione "distribute_tasks".
    # "distribute_tasks" restituisce una lista di Send(), che LangGraph mappa su "worker_node".
    workflow.add_conditional_edges("master", distribute_tasks, ["worker_node"])

    # Fan-in: Tutti i worker confluiscono nel Merger
    workflow.add_edge("worker_node", "merger")

    workflow.add_edge("merger", END)

    return workflow.compile()


# --- ESECUZIONE DI TEST (Main locale opzionale) ---
if __name__ == "__main__":
    # Esempio per testare il flusso isolatamente
    try:
        # Sostituisci con un path reale se vuoi testare
        with open("dataset_generated/stress_test_general/hospital_rules.json", "r") as f:
            rules_data = json.load(f)
    except FileNotFoundError:
        print("⚠️ File regole non trovato, uso lista vuota per test sintattico.")
        rules_data = []

    initial_state = {
        "user_instructions": "Generate hospital data",
        "original_rules": rules_data,
        "num_rows": 1000,
        "generated_code_snippets": [],
        "final_combined_code": ""
    }

    print("🚀 Avvio Pipeline Multi-Agente (Test Mode)...")
    app = build_parallel_graph()

    # Invoke
    result = app.invoke(initial_state)

    # Output
    print(f"\n✅ Pipeline completata.")
    if result.get("final_combined_code"):
        print("Script generato correttamente (primi 100 char):")
        print(result["final_combined_code"][:100])