import torch
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

MODEL_NAME = "google/pegasus-large"  
tokenizer = PegasusTokenizer.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PegasusForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

def text_summarization(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    
    #
    summary_ids = model.generate(**inputs, max_length=100, min_length=50, num_beams=2, early_stopping=True)
    
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
