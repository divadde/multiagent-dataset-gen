import pandas as pd
import json

def check_data_quality(csv_file_path, json_rules_path):
    # 1. Caricamento dei dati e delle regole
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Dataset caricato: {len(df)} righe.")
    except Exception as e:
        print(f"Errore nel caricamento del CSV: {e}")
        return

    try:
        with open(json_rules_path, 'r') as f:
            rules = json.load(f)
        print(f"Regole caricate: {len(rules)} regole trovate.\n")
    except Exception as e:
        print(f"Errore nel caricamento del JSON: {e}")
        return

    violations_count = 0

    # 2. Iterazione sulle regole
    for i, rule in enumerate(rules, 1):
        category = rule.get("category", "")
        targets = rule.get("target_columns", [])
        description = rule.get("description", "")

        print(f"--- Regola {i}: {description[:80]}... ---")

        # ---------------------------------------------------------
        # GESTIONE: 2. Functional Dependency (FD) Checking Rules
        # ---------------------------------------------------------
        if "Functional Dependency" in category:
            if len(targets) < 2:
                print(f"  [SKIP] Formato target non valido per FD: {targets}")
                continue

            # Convenzione: Ultima colonna = Dipendente, Altre = Determinanti
            determinants = targets[:-1]
            dependent = targets[-1]

            # Verifica esistenza colonne
            if not all(col in df.columns for col in targets):
                missing = [c for c in targets if c not in df.columns]
                print(f"  [ERRORE] Colonne mancanti nel CSV: {missing}")
                continue

            # Logica Pandas per FD:
            # Raggruppa per determinanti e conta i valori unici della dipendente
            # Se count > 1, c'è una violazione (stessi determinanti -> diversi dipendenti)
            violations = df.groupby(determinants)[dependent].nunique()
            invalid_groups = violations[violations > 1]

            if not invalid_groups.empty:
                print(f"  [FAIL] Violazione FD rilevata!")
                print(f"         Determinanti: {determinants} -> Dipendente: {dependent}")
                print(f"         Numero di gruppi violati: {len(invalid_groups)}")
                # Esempio di violazione
                example_idx = invalid_groups.index[0]
                print(f"         Esempio (Determinanti: {example_idx}):")
                if isinstance(example_idx, tuple):
                    query_str = " & ".join([f"`{col}` == {repr(val)}" for col, val in zip(determinants, example_idx)])
                else:
                    query_str = f"`{determinants[0]}` == {repr(example_idx)}"
                print(df.query(query_str)[targets].head(2).to_string(index=False))
                violations_count += 1
            else:
                print(f"  [OK] Regola FD soddisfatta.")

        # ---------------------------------------------------------
        # GESTIONE: 12. Aggregated Amount (Non-Unicità / Cardinalità)
        # ---------------------------------------------------------
        elif "Aggregated Amount" in category:
            # Assumiamo che per questo caso specifico stiamo controllando la non-unicità
            # se c'è una sola colonna target.
            if len(targets) == 1:
                col = targets[0]
                if col not in df.columns:
                    print(f"  [ERRORE] Colonna mancante: {col}")
                    continue

                total_rows = len(df)
                unique_vals = df[col].nunique()

                # La regola chiede che NON sia unico, quindi unique_vals < total_rows
                if unique_vals < total_rows:
                    print(f"  [OK] Regola di NON unicità soddisfatta.")
                    print(f"       Colonna '{col}': {unique_vals} valori unici su {total_rows} righe.")
                    print(f"       Tasso di duplicazione: {1 - (unique_vals / total_rows):.2%}")
                else:
                    print(f"  [FAIL] Violazione! La colonna '{col}' è univoca (Chiave Primaria).")
                    print(f"         Valori unici: {unique_vals} su {total_rows} righe.")
                    violations_count += 1
            else:
                print("  [INFO] Logica di aggregazione complessa non implementata in questo script base.")


        elif "Unique Key Constraint" in category:
            # Verifica che la combinazione delle colonne target sia unica
            if not all(col in df.columns for col in targets):
                print(f"  [ERRORE] Colonne mancanti: {targets}")
                continue

            # Conta duplicati sulla combinazione di colonne
            duplicates = df.duplicated(subset=targets, keep=False)
            num_duplicates = duplicates.sum()

            if num_duplicates > 0:
                print(f"  [FAIL] Trovate {num_duplicates} righe duplicate per la chiave {targets}!")
                print(df[duplicates].sort_values(by=targets).head(4).to_string(index=False))
                violations_count += 1
            else:
                print(f"  [OK] Nessun duplicato trovato per la chiave {targets}.")

    # Riepilogo finale
    if violations_count == 0:
        print("\n✅ SUCCESSO: Tutte le regole sono state soddisfatte!")
    else:
        print(f"\n❌ FALLIMENTO: Trovate {violations_count} violazioni delle regole.")

if __name__ == "__main__":
    #check_data_quality('0_rombo_dataset.csv', '0_rombo.json')
    #check_data_quality('1_rombo_ciclo.csv', '1_rombo_ciclo.json')
    #check_data_quality('2_it_asset_identity_cycle_dataset.csv', '2_it_asset_identity_cycles.json')
    #check_data_quality('3_directory_device_badge.csv', '3_directory_device_badge_cycles.json')

    check_data_quality('final_test.csv', 'final_test.json')