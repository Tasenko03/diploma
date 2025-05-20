import stanza

# Загружаем и кэшируем пайплайн один раз
_nlp = None

def get_stanza_pipeline(lang='ru'):
    global _nlp
    if _nlp is None:
        _nlp = stanza.Pipeline(lang=lang, use_gpu=True)
    return _nlp
