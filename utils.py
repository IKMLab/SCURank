import json, re
from util.anthropic import call_claude, batch_call_claude, call_grok
from util.gcloud import call_gemini, call_llama, call_mistral
from util.openai import call_openai, batch_call_openai
from tqdm import tqdm

def load_jsonl(pth: str):
    """
    Load a JSONL file into a list of dictionaries.
    """
    with open(pth, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data

def save_jsonl(data, pth: str):
    """
    Save a list of dictionaries to a JSONL file.
    """
    with open(pth, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def load_txt(path:str) -> list[str]:
    """
    Load a text file into a list of strings.
    Each line in the file becomes an item in the list.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = [line.strip() for line in f.readlines()]
    return data

def save_txt(path:str, data:list[str]):
    """
    Save a list of strings to a text file.
    """
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write("%s\n" % item)
        f.close()
    return None

def add_txt(path:str, text:str):
    """
    Append a string to a text file.
    """
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")
        f.close()
    return None


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def summary_cleanup(summary):
    # it represents that: the summary is below:\n\n[summary]
    if "\n\n" in summary:
        summary = ' '.join(summary.split('\n\n')[1:])
        summary = preprocess_text(summary)
    return summary

def call_llm(ins:str, content:str, model:str):
    if "claude" in model:
        return call_claude(ins, content, model)
    elif "grok" in model:
        return call_grok(ins, content, model)
    elif "gpt" in model:
        return call_openai(ins, content, model)
    elif "llama" in model:
        return call_llama(ins, content, model)
    elif "gemini" in model:
        return call_gemini(ins, content, model)
    elif "mistral" in model:
        return call_mistral(ins, content, model)
    else:
        raise ValueError(f"Model {model} not supported.")

def batch_call_llm(ins:str, contents:list, model:str, path:str, is_raw:bool = False):
    if "claude" in model:
        return batch_call_claude(ins, contents, model)
    elif "gpt" in model and not is_raw:
        return batch_call_openai(ins, contents, model, path)
    elif "llama" or "gemini" or "mistral" or "grok" or "gpt" in model:
        resps = []
        for c in tqdm(contents):
            resp = call_llm(ins, c, model)
            resp = summary_cleanup(resp)
            resps.append(resp)
            add_txt(f"{path}/{model}_resp.txt", resp)
        save_txt(f"{path}/{model}_resp", resps)
        print(f"Model: {model}, Summaries saved to {path}")
        return resps
    else:
        raise ValueError(f"Model {model} not supported.")