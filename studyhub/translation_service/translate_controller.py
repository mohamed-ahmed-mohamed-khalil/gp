import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
from torch.cuda.amp import autocast
from functools import lru_cache

# Cache models for reuse
@lru_cache(maxsize=3)  # Thread-safe caching
def load_model(src_lang, tgt_lang):
    """
    Loads and caches the translation model to avoid repeated loading.
    """
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model: {model_name} on {device}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        return tokenizer, model, device
    except Exception as e:
        raise RuntimeError(f"❌ Model loading failed: {e}")

def split_text_into_sentences(text):
    """
    Splits long text into sentences while keeping context intact.
    """
    return re.split(r'(?<=[.!?])\s+', text)  # Split based on punctuation

def translate_text(text, src_lang, tgt_lang, batch_size=8):
    """
    Translates long text efficiently by breaking it into smaller parts and using batching.
    """
    sentences = split_text_into_sentences(text)

    try:
        tokenizer, model, device = load_model(src_lang, tgt_lang)
    except RuntimeError as e:
        return str(e)  # Return error message

    translations = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        input_ids = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids.to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast():  # Corrected autocast usage
                output = model.generate(input_ids, max_length=256)  # Adjusted max_length
        
        translated_sentences = tokenizer.batch_decode(output, skip_special_tokens=True)
        translations.extend(translated_sentences)

    return " ".join(translations)
