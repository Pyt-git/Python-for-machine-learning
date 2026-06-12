import re

def clean_text(text): 
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

label_map = {
    "positive": 1, 
    "neutral": 0, 
    "negative": -1
}

def add_sample(df, text, label): 
    clean = clean_text(text)
    label_id = label_map[label]
    df.loc[len(df)] = [text, clean, label, label_id]
    return df
