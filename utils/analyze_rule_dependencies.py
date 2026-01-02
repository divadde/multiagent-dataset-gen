import networkx as nx
import json
import itertools
from collections import defaultdict


def analyze_rule_dependencies(rules):
    """
    Analizza le regole e raggruppa colonne e regole in cluster indipendenti.
    """
    G = nx.Graph()

    # Mappa per tenere traccia di quali regole appartengono a quali archi
    rule_map = defaultdict(list)

    # 1. Costruzione del Grafo
    for i, rule in enumerate(rules):
        targets = rule.get("target_columns", [])

        # Se la regola ha meno di 2 colonne, è un vincolo su singola colonna (nodo)
        if len(targets) == 1:
            G.add_node(targets[0])
            rule_map[frozenset([targets[0]])].append(rule)
            continue

        # Se la regola coinvolge più colonne, crea un arco (o una clique) tra loro
        # "Gli attributi sono nodi e le regole sono archi non orientati"
        for col1, col2 in itertools.combinations(targets, 2):
            G.add_edge(col1, col2)
            # Associamo la regola a questa connessione
            rule_map[frozenset([col1, col2])].append(rule)

    # 2. Individuazione delle Componenti Connesse (I Cluster)
    # Ogni componente è un gruppo di colonne che sono intrecciate dalle regole
    components = list(nx.connected_components(G))

    clusters = []

    for comp_idx, nodes in enumerate(components):
        nodes_list = list(nodes)
        cluster_rules = []

        # Recuperiamo tutte le regole che insistono ESCLUSIVAMENTE su questi nodi
        # (O che sono parte degli archi interni a questo cluster)
        seen_rules_ids = set()

        for rule in rules:
            targets = set(rule.get("target_columns", []))
            # Se tutte le colonne target sono nel cluster corrente, la regola appartiene al cluster
            if targets.issubset(nodes):
                # Usiamo la descrizione o l'indice come ID univoco per evitare duplicati
                rule_desc = rule.get("description")
                if rule_desc not in seen_rules_ids:
                    cluster_rules.append(rule)
                    seen_rules_ids.add(rule_desc)

        clusters.append({
            "cluster_id": comp_idx,
            "columns": nodes_list,
            "rules": cluster_rules,
            "complexity": len(cluster_rules)
        })

    return clusters


import networkx as nx
import itertools
from typing import List, Dict, Any


def get_clusters_from_rules(rules: List[Dict]) -> List[Dict]:
    """
    Costruisce il grafo delle dipendenze e restituisce una lista di cluster.
    Ogni cluster contiene: le regole, le colonne coinvolte e il 'peso' (numero di regole).
    """
    G = nx.Graph()

    # Mappa per associare ogni arco/nodo alle regole che lo generano
    # Usiamo l'indice della regola per tracciarla univocamente
    rule_map = {}

    for i, rule in enumerate(rules):
        targets = rule.get("target_columns", [])
        if not targets:
            continue

        # Aggiungiamo nodi e archi
        if len(targets) == 1:
            G.add_node(targets[0])
        else:
            # Crea una clique tra tutte le colonne target della regola
            for col1, col2 in itertools.combinations(targets, 2):
                G.add_edge(col1, col2)

        # Assegnamo questa regola a tutte le colonne coinvolte
        # (Semplificazione: associamo la regola al cluster che conterrà queste colonne)
        rule['id'] = i  # Assegnamo un ID temporaneo se non c'è

    # Trova le componenti connesse (i Cluster indipendenti)
    connected_components = list(nx.connected_components(G))

    clusters = []

    # Per ogni componente, raccogliamo le regole pertinenti
    for comp_idx, nodes in enumerate(connected_components):
        nodes_set = set(nodes)
        cluster_rules = []

        for rule in rules:
            targets = set(rule.get("target_columns", []))
            # Una regola appartiene al cluster se TUTTE le sue target columns sono nel cluster
            if targets.issubset(nodes_set):
                cluster_rules.append(rule)

        if cluster_rules:
            clusters.append({
                "cluster_id": comp_idx,
                "columns": list(nodes_set),
                "rules": cluster_rules,
                "rule_count": len(cluster_rules)
            })

    return clusters


def bin_packing_clusters(clusters: List[Dict], max_rules_per_agent: int = 8) -> List[Dict]:
    """
    Raggruppa i cluster in 'batches' in modo che la somma delle regole
    in ogni batch non superi max_rules_per_agent.
    Usa un approccio 'First Fit Decreasing' (Greedy).
    """
    # 1. Ordina i cluster dal più grande al più piccolo per ottimizzare il riempimento
    sorted_clusters = sorted(clusters, key=lambda x: x['rule_count'], reverse=True)

    batches = []

    for cluster in sorted_clusters:
        placed = False

        # Prova a inserire il cluster in un batch esistente
        for batch in batches:
            if batch['total_rules'] + cluster['rule_count'] <= max_rules_per_agent:
                batch['clusters'].append(cluster)
                batch['all_rules'].extend(cluster['rules'])
                batch['all_columns'].extend(cluster['columns'])
                batch['total_rules'] += cluster['rule_count']
                placed = True
                break

        # Se non entra in nessuno, crea un nuovo batch
        if not placed:
            batches.append({
                "batch_id": len(batches),
                "clusters": [cluster],  # Teniamo traccia dei cluster originali
                "all_rules": cluster['rules'],  # Lista piatta di tutte le regole
                "all_columns": cluster['columns'],  # Lista piatta di tutte le colonne
                "total_rules": cluster['rule_count']
            })

    return batches


def plan_generation_tasks(rules: List[Dict], max_rules: int = 8):
    """Funzione helper principale da chiamare nel nodo Master"""
    clusters = get_clusters_from_rules(rules)
    batches = bin_packing_clusters(clusters, max_rules_per_agent=max_rules)
    return batches


import networkx as nx
import itertools
import matplotlib.pyplot as plt  # <--- Aggiungi questo import
from typing import List, Dict, Any
from collections import defaultdict


# ... (Lascia invariate le funzioni esistenti: analyze_rule_dependencies, get_clusters_from_rules, etc.) ...

def visualize_attribute_dependencies(rules: List[Dict], output_path: str = "attribute_dependencies.png"):
    """
    Genera un grafo visivo delle dipendenze tra gli attributi basato sulle regole
    e lo salva come immagine PNG.
    """
    print(f"📊 Generazione grafico dipendenze in '{output_path}'...")

    G = nx.Graph()

    # 1. Costruzione del Grafo (Attributi = Nodi, Regole = Archi)
    for rule in rules:
        targets = rule.get("target_columns", [])
        if not targets:
            continue

        # Aggiungi i nodi (attributi)
        for col in targets:
            G.add_node(col)

        # Se la regola coinvolge più colonne, crea archi tra loro
        if len(targets) > 1:
            for col1, col2 in itertools.combinations(targets, 2):
                G.add_edge(col1, col2)

    # 2. Configurazione del Plot
    plt.figure(figsize=(14, 10))

    # Calcolo del layout (Spring layout distanzia i nodi in base agli archi)
    # k regola la distanza ottimale tra i nodi
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

    # Disegno dei nodi
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="lightblue", alpha=0.9)

    # Disegno degli archi
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.6, edge_color="gray")

    # Disegno delle etichette (nomi colonne)
    nx.draw_networkx_labels(G, pos, font_size=9, font_family="sans-serif", font_weight="bold")

    plt.title(f"Attribute Dependency Graph ({len(rules)} Rules, {G.number_of_nodes()} Attributes)", fontsize=16)
    plt.axis("off")  # Nasconde gli assi cartesiani

    # 3. Salvataggio
    try:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print("✅ Grafico salvato con successo.")
    except Exception as e:
        print(f"❌ Errore nel salvataggio del grafico: {e}")
    finally:
        plt.close()  # Chiude la figura per liberare memoria

# Esempio di utilizzo con il caricamento del file
if __name__ == "__main__":
    with open("../dataset_generated/stress_test_general/hospital_rules.json", "r") as f:
        rules_data = json.load(f)

    clusters = analyze_rule_dependencies(rules_data)

    print(f"Trovati {len(clusters)} cluster indipendenti.\n")
    for c in clusters:
        print(f"--- CLUSTER {c['cluster_id']} ---")
        print(f"Columns ({len(c['columns'])}): {c['columns']}")
        print(f"Rules Num: {c['complexity']}")
        print("-" * 20)