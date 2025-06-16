import json
import os
from google import genai
from dotenv import load_dotenv
import time # To handle potential rate limits
import requests
from groq import Groq

from pathinator import *

load_dotenv()

GEMINI_MODEL_NAME = "gemini-2.0-flash" # Or other suitable Gemini model
INPUT_PATH = dataset_subset # Path to your dataset file
OUTPUT_PATH = cleaning_results 
NUM_ITEM_TO_PROCESS = 3

api_key = os.getenv("GOOGLE_API_KEY")

# ----------------------------------------------- LLM Cleaning Functions -----------------------------------------------
def clean_with_gemini(client: genai.Client, ocr_text: str) -> str:
    """Cleans OCR text using Google Gemini (Client API style)."""
    if not ocr_text.strip():
        return ""

    try:
        # Initialize the client. Passing api_key explicitly is robust.
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Error initializing Gemini Client (genai.Client): {e}")
        print("This could be due to an invalid API key, network issues, or problems with the 'google-genai' library.")
        return f"[GEMINI_CLIENT_INIT_ERROR: {e}]"

    contents = f"""

**Titolo del Prompt:** 
Correttore OCR di Caratteri e Parole Spezzate

**Ruolo e Obiettivo Primario:**
Sei un sistema di pulizia OCR ultra-specifico. Il tuo unico compito è correggere caratteri palesemente errati e riunire parole spezzate dagli a capo. NON devi fare altro. NON DEVI UNIRE I PARAGRAFI.

**Regole Obbligatorie e Inviolabili:**

1.  **Correggi Solo Errori di Carattere e Parole Evidenti:** Rettifica solo errori di interpretazione dei singoli caratteri, sequenze o parole intere che sono chiaramente frutto di un errore OCR.
    *   **Esempi OBBLIGATORI:**
        *   `Kon` -> `Non`
        *   `Cilirgìa` -> `Ciliegia`
        *   `gTidò` -> `gridò`
        *   `shito` -> `stato`
        *   `Percliè` -> `Perchè`
        *   `Megnamc` -> `falegname` (Correggi parole intere solo se palesemente errate)
        *   `gTattandosi` -> `grattandosi`

2.  **Rimuovi Artefatti di Scansione:** Elimina caratteri e stringhe spuri che non hanno senso nel contesto.
    *   **Esempi OBBLIGATORI:**
        *   Rimuovi `pyyyyy^y^’^wggfy;` e simili.
        *   Rimuovi `[` , `]` , `^` se non fanno parte di una parola.
        *   `piacer^` -> `piacere`

3.  **Unisci Parole Spezzate (Solo con Trattino):** Ricomponi le parole divise da un trattino e un a capo (es. `fale-\ngname` -> `falegname`).

4.  **NON TOCCARE NULL'ALTRO (Regola Fondamentale):**
    *   **NON UNIRE PARAGRAFI O RIGHE.** Lascia tutti i `\n` esattamente come sono nel testo di input. Questa è la regola più importante.
    *   **NON TOCCARE LA PUNTEGGIATURA:** Lascia `.`, `,`, `!`, `?`, `:`, `;`, `—` invariati.
    *   **NON MODERNIZZARE PAROLE:** `maraviglioso`, `dètte`, `perchè` restano invariati.
    *   **NON RIMUOVERE RIGHE INTERE**, anche se sembrano didascalie. Questo compito spetta al secondo prompt.

**Formato di Output:**
Restituisci il testo con solo le correzioni di carattere, parola e artefatti. La struttura delle righe e dei paragrafi deve rimanere IDENTICA all'input.

OCR Text:
---
{ocr_text}
---
Cleaned Text:
"""
    response = client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=contents
    )
    #print(response.text)
    return response.text

def corrector_gemini(client: genai.Client, ocr_text: str) -> str:
    """Cleans OCR text using Google Gemini (Client API style)."""
    if not ocr_text.strip():
        return ""

    try:
        # Initialize the client. Passing api_key explicitly is robust.
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Error initializing Gemini Client (genai.Client): {e}")
        print("This could be due to an invalid API key, network issues, or problems with the 'google-genai' library.")
        return f"[GEMINI_CLIENT_INIT_ERROR: {e}]"

    contents = f"""

**RUOLO E OBIETTIVO:**
Sei un esperto filologo digitale e redattore specializzato nel restauro del testo "Le avventure di Pinocchio" di Collodi. Il testo che ricevi ha già subito una prima correzione a livello di carattere. Il tuo compito è ora finalizzare la struttura dei paragrafi e lo stile, aderendo in modo assoluto all'originale.

**ISTRUZIONI SEQUENZIALI E OBBLIGATORIE:**

1.  **Ricostruisci i Paragrafi:**
    *   Unisci le righe consecutive per formare paragrafi fluidi.
    *   Un paragrafo termina e ne inizia uno nuovo solo se c'è una riga vuota (doppio `\n`) o se una riga inizia con un trattino di dialogo (`—`).

2.  **Correggi il Numero di Capitolo:** Se il testo inizia con un numero romano palesemente errato come `IL`, correggilo in `II.`. Non modificare altri numeri romani corretti.

3.  **Rimuovi Didascalie e Rumore Residuo:**
    *   Elimina righe che sono chiaramente didascalie di immagini o ripetizioni fuori contesto. Esempio: rimuovi la riga `TJn vecchietto tutto arzillo, il quale aveva nome Geppetto.` se appare isolata prima del paragrafo che la contiene.
    *   Rimuovi eventuali stringhe di rumore residue.

**REGOLE DI PRESERVAZIONE ASSOLUTA (NON FARE):**
Questa è la sezione più importante. La violazione di queste regole invalida il risultato.

*   **NON MODERNIZZARE MAI:** È severamente proibito alterare l'ortografia o il lessico originale.
    *   `perchè` **DEVE rimanere** `perchè` (NON `perché`).
    *   `maraviglioso` **DEVE rimanere** `maraviglioso`.
    *   `trasfigurito` **DEVE rimanere** `trasfigurito` (NON `trasfigurato`).
    *   `messe` **DEVE rimanere** `messe` (NON `mise`).
    *   `sbatacchiarlo` **DEVE rimanere** `sbatacchiarlo` (NON `sbatterlo`).
    *   `su i` (separato) **DEVE rimanere** `su i` (NON `sui`).
    *   `corbello` **DEVE rimanere** `corbello` (NON `corbèllo`).

*   **NON ALTERARE LA STRUTTURA DELLA FRASE:** Non aggiungere parole (es. `E` all'inizio di una frase come in `E rimettiamoci a lavorare`), non rimuoverle e non cambiarne l'ordine.

*   **NON ALTERARE LA PUNTEGGIATURA:** Mantieni la punteggiatura originale (`—`, `....`, `!...`, `?...`). L'unica eccezione è correggere un trattino semplice `-` in un trattino lungo `—` all'inizio di un dialogo, se necessario.

**FORMATO DI OUTPUT:**
Restituisci solo ed esclusivamente il testo finale, pulito e corretto secondo queste rigide regole. Non aggiungere commenti o spiegazioni.

OCR Text:
---
{ocr_text}
---
Cleaned Text:
"""
    response = client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=contents
    )
    #print(response.text)
    return response.text

#--------------------------------------------------------------------------------------------------------------------------------

def clean_with_groq(client: Groq, ocr_text: str, model_id: str) -> str:
    """Cleans OCR text using a model from the Groq API."""
    if not ocr_text.strip():
        return ""

    # This chat-based prompt works well for modern instruct models like Llama3 and Mixtral
    messages = [
        {
            "role": "system",
            "content": """
            
**Titolo del Prompt:** 
Correttore OCR di Caratteri e Parole Spezzate

**Ruolo e Obiettivo Primario:**
Sei un sistema di pulizia OCR ultra-specifico. Il tuo unico compito è correggere caratteri palesemente errati e riunire parole spezzate dagli a capo. NON devi fare altro.
NON SALTARE NESSUNA PAROLA

**Regole Obbligatorie e Inviolabili:**

1.  **Correggi Solo Errori di Carattere Evidenti:** Rettifica solo errori di interpretazione dei singoli caratteri o piccole sequenze.
    *   **Esempi OBBLIGATORI:**
        *   `Kon` -> `Non`
        *   `Cilirgìa` -> `Ciliegia`
        *   `gTidò` -> `gridò`
        *   `troi^pa` -> `troppa`
        *   `Ohe` -> `Che`
        *   `piii` -> `più`
        *   `shito` -> `stato`
        *   `piacer^` -> `piacere`
        *   `Percliè` -> `Perchè`
        *   `gU lacesse V elemosina` -> `gli facesse l'elemosina`
        *   `TJn` -> `Un`
        *   Rimuovi caratteri spuri come `[` , `]` , `^` che non fanno parte di una parola.

2.  **Rimuovi Artefatti di Scansione:** Elimina caratteri spuri che non hanno senso nel contesto, come parentesi quadre isolate o simboli errati.
    *   Esempio: `nulla, [il gran nulla` -> `nulla, il gran nulla`. `piacer^` -> `piacere`.

3.  **Unisci Parole Spezzate (Solo con Trattino):** Ricomponi le parole divise da un trattino e un a capo (es. `fale-\ngname` -> `falegname`).

4.  **NON TOCCARE NULL'ALTRO:**
    *   **NON unire paragrafi.** Lascia tutti i `\n` esattamente come sono.
    *   **NON rimuovere righe.**
    *   **NON TOCCARE LA PUNTEGGIATURA:** Lascia `.`, `,`, `!`, `?`, `:`, `;` invariati.
    *   **NON modernizzare parole** (`maraviglioso`, `dètte` restano invariati per ora).

**Formato di Output:**
Restituisci il testo con solo le correzioni di carattere e artefatti. Il resto deve essere identico all'input.""",


        },
        {
            "role": "user",
            "content": f"OCR Text:\n---\n{ocr_text}\n---\nCleaned Text:",
        }
    ]

    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model_id,
            temperature=0.1,  # Lower temperature for more deterministic output
            max_tokens=2048,
        )
        cleaned_text = chat_completion.choices[0].message.content.strip()
        return cleaned_text
    except Exception as e:
        return f"[GROQ_API_ERROR: {e}]"
    

def corrector_groq(client: Groq, ocr_text: str, model_id: str) -> str:
    """Cleans OCR text using a model from the Groq API."""
    if not ocr_text.strip():
        return ""

    # This chat-based prompt works well for modern instruct models like Llama3 and Mixtral
    messages = [
        {
            "role": "system",
            "content": """
            
### RUOLO E OBIETTIVO
Sei un esperto filologo digitale e redattore specializzato nel testo "Le avventure di Pinocchio" di Collodi. Il testo che ricevi ha già subito una correzione a livello di carattere. Il tuo compito è ora quello di finalizzare la struttura, la punteggiatura e lo stile secondo l'originale.
NON SALTARE NESSUNA PAROLA

### ISTRUZIONI SPECIFICHE E SEQUENZIALI

1.  **Sillabazione:** Prima di tutto, unisci le parole spezzate da un trattino e un a capo. Esempio: `fale-\ngname` -> `falegname`.

2.  **Correggi Numero di Capitolo (con Logica):**
    *   Controlla il numero romano all'inizio del testo. Se è palesemente errato rispetto al contesto (es. `IL`), correggilo. Se è un numero romano valido (es. `V`), assicurati solo che sia seguito da un punto. NON cambiarlo se è già corretto.
    *   Esempio di correzione: `IL` -> `II.`. Esempio di mantenimento: `V` -> `V.`.

3.  **Rifinitura Grammaticale e Lessicale:**
    *   Correggi errori grammaticali sottili. Esempi: `appetito dei ragazzi` -> `appetito nei ragazzi`.
    *   Correggi gli accenti errati su parole comuni. Esempio OBBLIGATORIO: `si dètte` -> `si dette`.
    *   Assicurati che la `e` verbale sia `è`.

4.  **Punteggiatura Logica nei Dialoghi e nella Narrazione:**
    *   Analizza il contesto per usare la punteggiatura corretta. Presta la massima attenzione ai dialoghi.
    *   Una frase interrogativa deve terminare con `?` o `?…`. Una esclamazione con `!` o `!...`.
    *   Esempio OBBLIGATORIO: `Ne farò una frittata!...` -> `Ne farò una frittata?…`.
    *   Sistema la spaziatura e le virgole per migliorare la leggibilità, rispettando lo stile originale. Esempio: `fame, e la fame, dal vedere al non vedere si` -> `fame, e la fame, dal vedere al non vedere, si`.
    *   Mantieni le maiuscole:** `piatto!… O non sarebbe` deve rimanere con la `O` maiuscola.

5.  **Stile di Collodi (Regole di Conservazione):**
    *   **NON confondere `se` e `sé`:** `se` non è mai accentato tranne quando è pronome tonico (`fa da sé`). Correggi eventuali `sè` errati.
    *   **Mantieni `perchè`** con l'accento grave. 
    *   **NON MODERNIZZARE:** Mantieni `maraviglioso`, `formicole`, `costì`, `bicchier`.

### FORMATO DI OUTPUT
Restituisci solo ed esclusivamente il testo finale, pulito e corretto.""",


        },
        {
            "role": "user",
            "content": f"OCR Text:\n---\n{ocr_text}\n---\nCleaned Text:",
        }
    ]

    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model_id,
            temperature=0.1,  # Lower temperature for more deterministic output
            max_tokens=2048,
        )
        cleaned_text = chat_completion.choices[0].message.content.strip()
        return cleaned_text
    except Exception as e:
        return f"[GROQ_API_ERROR: {e}]"

#--------------------------------------------------------------------------------------------------------------------------------

def main():
    """
    main function to create a clean text from ocr text using different LLMs
    this is actually no more nedded because the main file take the function from cleaner_LLM.py. but could be usefull if we want to create a separate file witout run the main file
    """
  
    try:
        with open(INPUT_PATH, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
        print(f"Successfully loaded {len(data_dict)} items from '{INPUT_PATH}'")
    except FileNotFoundError:
        print(f"Error: Input file not found at '{INPUT_PATH}'")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{INPUT_PATH}'. Please check its format.")
        return
    
    # Initialize Gemini client once
    try:
        gemini_client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Could not initialize Gemini Client: {e}")
        return

    list_of_items = list(data_dict.values())
    print(f"Total items (segments with ocr/clean pairs) to process: {len(list_of_items)}")

    subset_to_process = list_of_items[:NUM_ITEM_TO_PROCESS]
    print(f"\nProcessing {len(subset_to_process)} items from the dataset...")

    results = []

    # --- 3. Process Data ---
    for i, item in enumerate(subset_to_process):
        print(f"\n--- Processing item {i+1}/{len(subset_to_process)} ---")

        ocr_text = item['ocr']
        ground_truth_clean_text = item['clean']

        print(f"OCR Text (first 200 chars):\n{ocr_text[:200]}{'...' if len(ocr_text) > 200 else ''}")

        print("Cleaning with Gemini...")
        gemini_cleaned_text = clean_with_gemini(gemini_client, ocr_text)
        if "[GEMINI_" in gemini_cleaned_text: # Check if an error placeholder was returned
            print(f"Gemini cleaning failed for item {i+1}. Returned: {gemini_cleaned_text}")
        else:
            print(f"Gemini Cleaned Text (first 200 chars):\n{gemini_cleaned_text[:200]}{'...' if len(gemini_cleaned_text) > 200 else ''}")

    # To avoid hitting rate limits, we can add a sleep here.
        time.sleep(1)

        results.append({
            "original_ocr": ocr_text,
            "ground_truth": ground_truth_clean_text,
            "gemini_cleaned": gemini_cleaned_text,
            
        })

# Save Results to JSON File 
    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as outfile:
            json.dump(results, outfile, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to {OUTPUT_PATH}")
    except IOError as e:
        print(f"\nError saving results to {OUTPUT_PATH}: {e}")


if __name__ == "__main__":
    main()


"""

**Titolo del Prompt:** 
Correttore Iper-Letterale di Testo OCR

**Ruolo e Obiettivo Primario:**
Sei un sistema di pulizia del testo iper-letterale, specializzato nell'elaborazione di output OCR da testi italiani, in particolare "Le avventure di Pinocchio". 
Il tuo unico e obbligatorio compito è correggere errori di scansione palesemente errati a livello di carattere e di formattazione, 
preservando in modo assoluto e letterale il testo originale, comprese le sue particolarità ortografiche.

**Regole Obbligatorie e Inviolabili (L'aderenza deve essere totale):**

1.  **Correggi Esclusivamente Errori OCR:** La tua funzione principale è rettificare evidenti errori di interpretazione dei caratteri da parte dell'OCR. 
Esempi: `Kon` deve diventare `Non`, `Cilirgìa` deve diventare `Ciliegia`, `colica` deve diventare `colpa`, `Piuoccliio` deve diventare `Pinocchio`, 
`e'` deve diventare `c'è`, `gii` deve diventare `gli`.

2.  **NON Modificare Parole Già Corrette:** È severamente proibito alterare o modernizzare parole che siano già scritte correttamente, 
anche se in una forma letteraria o oggi meno comune. Esempio: `maraviglioso` deve rimanere `maraviglioso` (e non diventare `meraviglioso`), 
`perchè` deve rimanere `perchè` (e non `perché`), `egli` deve rimanere `egli` (e non `lui`).

3.  **NON Alterare la Struttura della Frase:** Non aggiungere, eliminare o riordinare alcuna parola. La sequenza di parole deve rimanere identica all'originale.

4.  **Preserva la Punteggiatura:** Lascia ogni segno di punteggiatura (virgole, punti, punti e virgola, ecc.) esattamente dov'è. 
Non aggiungerne di nuovi, non rimuovere quelli esistenti. L'unica eccezione è normalizzare sequenze multiple di punti (es. `....`) in una forma standard (`...` o `... ―`).

5.  **Ricostruisci i Paragrafi:** Unisci le righe che sono state spezzate artificialmente dall'OCR per formare paragrafi fluidi e coerenti. 
Esempio: `...si met- \ntono nelle stufe...` deve diventare `...si mettono nelle stufe...`.

6.  **Mantieni l'Integrità Morfologica:** Se una parola errata è un plurale, la sua versione corretta deve essere anch'essa plurale. 
Se è un verbo coniugato, deve mantenere la sua coniugazione originale.

7.  **Elimina il Rumore di Scansione:** Rimuovi caratteri spuri, frammenti illeggibili o artefatti della scansione che non hanno senso nel contesto (es. `pyyyyy^y^'^wggfy;`).

8.  **Rimuovi Didascalie Intrusive:** Ignora e rimuovi didascalie di immagini o frammenti di testo che sono chiaramente estranei e interrompono il flusso narrativo principale.

9.  **Formato di Output:** Restituisci esclusivamente il blocco di testo corretto. Non includere spiegazioni, commenti, annotazioni, 
o qualsiasi testo che non faccia parte del contenuto pulito.

**Esempio di Esecuzione:**

**Input:**
`I. \nComo  andò  che  Maestro  Ciliegia,  Megnamc \ntrovò  un  pezzo  di  legno  che  piangeva  e  rideva  come  un  bambino. 
\n—  C'era  una  volta.... \n—  Un  re!  -diranno  subito  i  miei  piccoli  lettori. \n—  Ko,  ragazzi,  avete  sbagliato.  
C'era  una  volta \nun  pezzo  di  legno. \nKon  era  un  legno  di  lusso,  ma  un  semplice \npezzo  da  catasta...`

**Output Atteso (e unico output accettabile):**
`I.\n\nCome andò che Maestro Ciliegia, falegname\ntrovò un pezzo di legno che piangeva e rideva come un bambino.
\n\n— C’era una volta....\n— Un re! — diranno subito i miei piccoli lettori.\n— No, ragazzi, avete sbagliato. 
C’era una volta un pezzo di legno.\nNon era un legno di lusso, ma un semplice pezzo da catasta...`

**Conclusione Imperativa:**
Esegui solo ed esclusivamente le correzioni specificate. Non deviare, non interpretare, non aggiungere nulla che non sia esplicitamente richiesto da queste regole. 
La tua aderenza a questo protocollo deve essere letterale e assoluta."""