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

    contents = f"""Correggi questo testo OCR in italiano tratto da un libro: separa i paragrafi come in un’edizione classica e correggi parole errate SENZA MODERNIZZARE lo stile. 
    Le parole dovranno essere SOLO in italiano e dovranno rispecchiare fedelmente il significato supposto dell'errore ocr (es. Eimettiamoci diventa Rimettiamoci, Megnamc diventa falegname, Ohé diventa Che, AYRAHKO diventa AVRANNO).
    !!!IMPORTANTISSIMO, NON CORREGGERE MAI PAROLE GIA' GIUSTE!!!
    Nelle parole da correggere mantieni le forme d’epoca come “colla lingua”, “costì”, “maraviglia”, “formicole” ecc. 
    Fai attenzione a parole che sembrano d'epoca che in realtà sono solo errori dell'ocr (cridò diventa gridò)
    NON AGGIUNGERE NIENTE E NON ELIMINARE NULLA, RISPETTA FEDELMENTE IL TESTO! 
    Eliminazione consentita solo in caso di palesi errori di codificazione ocr tipo "f:^  ’ì", "idf2 ?d 3ili" o "283900".
    RISPETTA SEMPRE GLI ACCENTI E LA PUNTEGGIATURA (non li cambiare!). Presta MASSIMA ATTENZIONE alle parole con apostrofi o accenti.
    ATTENZIONE AI NOMI PROPRI DEI PERSONAGGI LA CUI PRIMA LETTERA è IN MAIUSCOLO.
    
    Restituisci SOLO e SOLAMENTE il testo pulito.

...
Esegui la correzione del seguente testo OCR:
---
{ocr_text}
---
Testo corretto:
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

    contents = f"""Rifinisci il seguente testo OCR precedentemente corretto da un altro LLM modificando SOLAMENTE le parole senza senso ed eliminando parole sbagliate o caratteri 
    che non hanno un significato legato alla frase.
    !!!IMPORTANTISSIMO, NON CORREGGERE MAI PAROLE GIA' GIUSTE!!!
    Nelle parole da correggere mantieni le forme d’epoca come “colla lingua”, “costì”, “maraviglia”... verificandone però l'esistenza, se non esistono scrivi la parola nel linguaggio attuale.
    RISPETTA SEMPRE GLI ACCENTI E LA PUNTEGGIATURA (non li cambiare!). 
    Ogni inizio capitolo inizia con un numero romano tipo “XV.\n\n” (es. IL diventa II, IH diventa III ecc...). 

    Restituisci SOLO e SOLAMENTE il testo pulito.

Esegui la correzione del seguente testo OCR prepulito:
---
{ocr_text}
---
Testo corretto:
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
            "content": """Correggi questo testo OCR in italiano tratto da un libro: separa i paragrafi come in un’edizione classica e correggi parole errate SENZA MODERNIZZARE lo stile. 
    Le parole dovranno essere SOLO in italiano e dovranno rispecchiare fedelmente il significato supposto dell'errore ocr (es. Eimettiamoci diventa Rimettiamoci, Megnamc diventa falegname, Ohé diventa Che, AYRAHKO diventa AVRANNO).
    !!!IMPORTANTISSIMO, NON CORREGGERE MAI PAROLE GIA' GIUSTE!!!
    Nelle parole da correggere mantieni le forme d’epoca come “colla lingua”, “costì”, “maraviglia”, “formicole” ecc. 
    Fai attenzione a parole che sembrano d'epoca che in realtà sono solo errori dell'ocr (cridò diventa gridò)
    NON AGGIUNGERE NIENTE E NON ELIMINARE NULLA, RISPETTA FEDELMENTE IL TESTO! 
    Eliminazione consentita solo in caso di palesi errori di codificazione ocr tipo "f:^  ’ì", "idf2 ?d 3ili" o "283900".
    RISPETTA SEMPRE GLI ACCENTI E LA PUNTEGGIATURA (non li cambiare!). Presta MASSIMA ATTENZIONE alle parole con apostrofi o accenti.
    ATTENZIONE AI NOMI PROPRI DEI PERSONAGGI LA CUI PRIMA LETTERA è IN MAIUSCOLO.
    
    Restituisci SOLO e SOLAMENTE il testo pulito."""

        },
        {
            "role": "user",
            "content": f"Esegui la correzione del seguente testo OCR::\n---\n{ocr_text}\n---\nTesto corretto:",
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
            "content": """Rifinisci il seguente testo OCR precedentemente corretto da un altro LLM modificando SOLAMENTE le parole senza senso ed eliminando parole sbagliate o caratteri 
    che non hanno un significato legato alla frase.
    !!!IMPORTANTISSIMO, NON CORREGGERE MAI PAROLE GIA' GIUSTE!!!
    Nelle parole da correggere mantieni le forme d’epoca come “colla lingua”, “costì”, “maraviglia”... verificandone però l'esistenza, se non esistono scrivi la parola nel linguaggio attuale.
    RISPETTA SEMPRE GLI ACCENTI E LA PUNTEGGIATURA (non li cambiare!).  
    Ogni inizio capitolo inizia con un numero romano tipo XV.\n\n (es. IL diventa II, IH diventa III ecc...). 

    Restituisci SOLO e SOLAMENTE il testo pulito.""",


        },
        {
            "role": "user",
            "content": f"Esegui la correzione del seguente testo OCR prepulito:\n---\n{ocr_text}\n---\nTesto corretto:",
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


"""
    Hai il compito di ripulire un testo italiano ottenuto tramite OCR, il quale può contenere:
    *  errori ortografici (es. "lamj)i", "veccliio", "darrai"),
    caratteri errati o fuori posto (es. “<x mi V» /■ ,”),
    lettere mancanti o sostituite (es. “P uscio” invece di “l’uscio”),
    punteggiatura danneggiata o mancante,
    spazi errati o line-breaks casuali,
    errori nei paragrafi.
    
    Il tuo obiettivo è correggere accuratamente il testo restando fedele alla grafia, al lessico e alla sintassi originali del tempo (non modernizzare). 
    Inoltre, devi identificare e separare i paragrafi coerentemente, come avverrebbe in un'edizione classica stampata.
    ⚠️ Mantieni le forme linguistiche d’epoca come “colla lingua”, “costì”, “sonare”, “uscio”, ecc.
    
    Esempio:

    Testo OCR (input):
    \nPareva  il  paese  dei  morti. \nAllora  Pinocchio,  i)reso  dalla  disperazione  e \ndalla  fame,  si  attaccò  al  campanello  d'una  casa,  e \ncominciò  a  sonare  a  distesa,  dicendo  dentro  di  sé  : \n—  Qualcuno  si  affaccerà.  — \n\nDifatti  si  affacciò  un  veccliio,  col  berretto  da \nuotte  in  capo,  il  quale  gridò  tutto  stizzito: \n—  Ohe  cosa  volete  a  quest'ora! \n\nHi  \"il     <x  mi  V»  /■ , \n\nTornò  a  casa  bagnato  come  un  pulcino.... \n\n—  Ohe  mi  fareste  il  piacere  di  darrai  un  po'  di \npane?

    Testo pulito (output):
    Pareva il paese dei morti.\nAllora Pinocchio, preso dalla disperazione e dalla fame, si attaccò al campanello d’una casa, e cominciò a suonare a distesa, dicendo dentro di sè:\n— Qualcuno si affaccerà. —\nDifatti si affacciò un vecchio, col berretto da notte in capo, il quale gridò tutto stizzito:\n— Che cosa volete a quest’ora?\nTornò a casa bagnato come un pulcino...\n\n— Che mi fareste il piacere di darmi un po’ di pane?"""