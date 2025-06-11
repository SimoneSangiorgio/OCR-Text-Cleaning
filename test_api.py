import requests
import json


#MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_ID = "deepseek-ai/DeepSeek-R1-0528"

HF_TOKEN = "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" 
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"


headers = {"Authorization": f"Bearer {HF_TOKEN}"}
prompt = "Quanto fa 2+2? Ti voglio sintetico"

payload = {
    "inputs": prompt,
    "parameters": {
        "temperature": 0.7,
        "max_new_tokens": 200,
        "top_p": 0.95,
        "return_full_text": False
    }
}

response = requests.post(API_URL, headers=headers, json=payload)
print(response.json())


# --- FUNZIONE ---
'''def query(payload):
    print(f"Sto inviando una richiesta POST a: {API_URL}")
    response = requests.post(API_URL, headers=headers, json=payload)
    return response

# --- ESECUZIONE ---
response = query(payload)
status_code = response.status_code
response_json = None

try:
    response_json = response.json()
except json.JSONDecodeError:
    print("La risposta non è in formato JSON.")

print(f"\n--- DEBUG INFO ---")
print(f"Status Code: {status_code}")
print(f"Risposta ricevuta: {response.text}")
print("--------------------")

# --- INTERPRETAZIONE DEL RISULTATO ---
if status_code == 200:
    print("\n✅ SUCCESSO! Risposta del modello:")
    # Estrae e stampa solo il testo generato
    generated_text = response_json[0].get('generated_text', 'Testo non trovato nella risposta.')
    print(generated_text)

elif status_code == 503 and response_json and 'estimated_time' in response_json:
    wait_time = response_json['estimated_time']
    print(f"\n⏳ ATTESA: Il modello si sta caricando. È normale.")
    print(f"Tempo di attesa stimato: {wait_time:.1f} secondi. Riprova tra poco.")
    
elif status_code == 401:
     print(f"\n❌ ERRORE: Non autorizzato (Unauthorized). Il tuo token HF_TOKEN è sbagliato o non valido.")

else:
    print(f"\n❌ ERRORE INASPETTATO (Codice: {status_code}). Controlla il testo della risposta sopra.")'''