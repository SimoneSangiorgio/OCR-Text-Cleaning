import re
from spellchecker import SpellChecker
from pathinator import dictionary

# 🔹 Carica dizionario personalizzato da index.dic
def load_custom_dictionary(path=dictionary):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    custom_dict = set()
    for line in lines:
        line = line.strip()
        if not line or line.startswith('/'):  # ignora commenti e vuoti
            continue
        word = re.split(r'[\/\s]', line)[0]  # prende solo la parola prima di "/OT..."
        custom_dict.add(word.lower())
    return custom_dict

# 🔹 Inizializza dizionario e correttore
spell = SpellChecker(language='it')
custom_dictionary = load_custom_dictionary(dictionary)
spell.word_frequency.load_words(custom_dictionary)

# 🔹 Funzione aggiornata
def correct_spelling(text):
    """
    Corregge l'ortografia del testo, ignorando:
    - Punteggiatura e spazi.
    - Parole che iniziano con una lettera maiuscola.
    - Parole presenti nel dizionario personalizzato.
    """
    tokens = re.findall(r'\w+|[^\w\s]|\s', text, re.UNICODE)
    corrected_tokens = []

    for token in tokens:
        # Condizione 1: Non toccare punteggiatura, spazi o numeri
        if not token.isalpha():
            corrected_tokens.append(token)
            continue

        # Condizione 2: Non correggere parole che iniziano con maiuscola (es. nomi propri)
        if token[0].isupper():
            corrected_tokens.append(token)
            continue
            
        # Condizione 3: Non correggere parole presenti nel dizionario personalizzato
        if token.lower() in custom_dictionary:
            corrected_tokens.append(token)
            continue

        # Se nessuna delle condizioni sopra è vera, prova a correggere la parola
        suggestion = spell.correction(token)
        corrected_tokens.append(suggestion if suggestion else token)

    return ''.join(corrected_tokens)
def clean_symbols(text):
    text = re.sub(r"[•–“”’‘]", "'", text)
    text = re.sub(r"[^\w\s,.!?'\-]", '', text)
    return text

def fix_ocr_specific_errors(text):
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)      # 1 234 -> 1234
    text = re.sub(r'-\n', '', text)                   # word break
    text = re.sub(r'\bl\b', 'i', text)                # 'l' as 'i'
    return text


def replacement_rules(text):
    REPLACEMENT_RULES = {
        "-\n\n": "",
        "-\n": "",
        "- \n": "",
        "  ": " ",      
        "é": "è",       
        "- ": "",       
        "'": "’",
        #"…": ". . ."      
    }
    
    for old, new in REPLACEMENT_RULES.items():
        text = text.replace(old, new)
        
    text = text.replace("  ", " ")
    return text

def full_preclean(text):
    text = replacement_rules(text)
    text = correct_spelling(text)
    return text

'''ocr_string = "IH. \nGeppetto,  tornato  a  casa,  comincia  subito  a  fabbricarsi  il  bu- \nrattino e  gli  mette  il  nome  di  Pinocchio.  Prime  monellerie \ndel  burattino. \nLa  casa  di  Geppetto  era  una  stanzina  terrena, \nche  pigliava  luce  da  un  sottoscala.  La  mobilia \nnon  poteva  esser  più  semplice:  una  seggiola  cat- \ntiva, un  letto  poco  buono  e  un  tavolino  tutto \nrovinato.  Nella  parete  di  fondo  si  vedeva  un \ncaminetto  col  fuoco  acceso  ;  ma  il  fuoco  era \ndipinto,  e  accanto  al  fuoco  e'  era  dipinta  un  pen- \ntola che  bolliva  allegramente  e  mandava  fuori \nuna  nuvola  di  fumo,  che  pareva  fumo  davvero. \nAppena  entrato  in  casa,  Geppetto  prese  subito \ngli  arnesi  e  si  pose  a  intagliare  e  a  fabbricare \nil  suo  burattino. \n—  Che  nome  gli  metterò  ?  —  disse  fra  sé  e \nsé.  —  Lo  voglio  chiamar  Pinocchio.  Questo  nome \ngli  porterà  fortuna.  Ho  conosciuto  una  famiglia \nintera  di  Pinocchi:  Pinocchio  il  padre,  Pinocchia \nla  madre  e  Pinocchi  i  ragazzi,  e  tutti  se  la  pas- \nsavano bene.  Il  più  ricco  di  loro  chiedeva  l' ele- \nmosina?^ \n\nQuando  ebbe  trovato  il  nome  al  suo  burattino, \nallora"
cleaned = full_preclean(ocr_string)

print("OCR originale:\n", ocr_string)
print("\nTesto dopo pre-cleaning:\n", cleaned)'''