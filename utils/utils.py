def extract_code_from_message(message):
    # 1. Recupera il contenuto (può essere stringa o lista di blocchi)
    content = message.content

    raw_text = ""

    # Se è una lista (nuovo formato GPT-5/Reasoning), cerca il blocco 'text'
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get('type') == 'text':
                raw_text = block.get('text', '')
                break
    # Se è già una stringa (formato classico)
    elif isinstance(content, str):
        raw_text = content

    # 2. Pulizia dei tag Markdown (```python ... ```)
    # Rimuove ```python all'inizio e ``` alla fine, e spazi bianchi extra
    clean_code = raw_text.replace("```python", "").replace("```", "").strip()

    return clean_code
