def extract_code_from_message(message_or_str):
    """
    Estrae il codice Python pulito da un messaggio LLM, gestendo vari formati di input
    (Stringa pura, AIMessage, Dizionario o Lista di blocchi).
    """
    content = ""

    # 1. Normalizzazione dell'input in 'content'
    if isinstance(message_or_str, str):
        content = message_or_str
    elif isinstance(message_or_str, dict):
        content = message_or_str.get('content', '')
    elif hasattr(message_or_str, 'content'):
        content = message_or_str.content
    else:
        content = str(message_or_str)

    raw_text = ""

    # 2. Gestione del contenuto (Stringa o Lista di blocchi per modelli Reasoning)
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get('type') == 'text':
                raw_text = block.get('text', '')
                break
        # Fallback se non trova blocchi text
        if not raw_text:
            raw_text = str(content)
    else:
        raw_text = str(content)

    # 3. Pulizia dei tag Markdown
    clean_code = raw_text.replace("```python", "").replace("```", "").strip()

    return clean_code
