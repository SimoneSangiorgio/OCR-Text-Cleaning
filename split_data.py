import json
from pathinator import *

# Path to the dataset file
SUBSET_SIZE = 35

input_ocr_path = dataset_path / "original_ocr.json"
input_clean_path = dataset_path / "cleaned.json"
output_subset_path = dataset_subset

subset_data = {}

subset_data = {}

try:
    # Carica i file JSON di OCR e testo pulito
    with open(input_ocr_path, 'r', encoding='utf-8') as f_ocr:
        ocr_data = json.load(f_ocr)

    with open(input_clean_path, 'r', encoding='utf-8') as f_clean:
        clean_data = json.load(f_clean)

    print(f"Caricate {len(ocr_data)} voci da {input_ocr_path}")
    print(f"Caricate {len(clean_data)} voci da {input_clean_path}")

    # 1. Estrai le chiavi, convertile in interi per l'ordinamento numerico
    #    Questo risolve il problema dell'ordinamento alfabetico ("10" prima di "2")
    keys_as_integers = [int(k) for k in ocr_data.keys()]

    # 2. Ordina le chiavi numericamente
    sorted_keys = sorted(keys_as_integers)
    total_chapters = len(sorted_keys)
    print(f"Trovate e ordinate {total_chapters} chiavi/capitoli totali.")

    # 3. Prendi solo il numero di voci specificato da SUBSET_SIZE
    if SUBSET_SIZE > total_chapters:
        print(f"Attenzione: Richieste {SUBSET_SIZE} voci, ma ne sono disponibili solo {total_chapters}. Verranno usate tutte le voci disponibili.")
        keys_for_subset = sorted_keys
    else:
        # Applica il slicing per ottenere solo il numero desiderato di chiavi
        keys_for_subset = sorted_keys[:SUBSET_SIZE]

    print(f"Creazione di un subset con le prime {len(keys_for_subset)} voci ordinate...")

    # 4. Itera attraverso le chiavi ordinate e popola il dizionario del subset
    for key_int in keys_for_subset:
        key_str = str(key_int)  # Riconverti in stringa per accedere ai dizionari originali

        if key_str in ocr_data and key_str in clean_data:
            subset_data[key_str] = {
                "ocr": ocr_data[key_str],
                "clean": clean_data[key_str]
            }
        else:
            print(f"Attenzione: La chiave '{key_str}' non è stata trovata in entrambi i file. Viene saltata.")

    # 5. Salva il nuovo dizionario del subset in un file JSON
    with open(output_subset_path, 'w', encoding='utf-8') as f_out:
        json.dump(subset_data, f_out, indent=2, ensure_ascii=False)

    print(f"\nOperazione completata con successo.")
    print(f"Creato un subset con {len(subset_data)} voci ordinate.")
    print(f"File salvato in: {output_subset_path}")

except FileNotFoundError as e:
    print(f"ERRORE: Il file {e.filename} non è stato trovato.")
except json.JSONDecodeError:
    print("ERRORE: Impossibile decodificare il JSON. Controlla la sintassi dei file di input.")
except ValueError:
    print("ERRORE: Una delle chiavi nel file JSON non è un numero valido. Controlla i file di input.")
except Exception as e:
    print(f"Si è verificato un errore inaspettato: {e}")