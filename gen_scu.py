from utils import call_llm, batch_call_llm
import os

def load_txt(path:str):
    with open(path, "r") as f:
        data = f.readlines()
    return [d.strip() for d in data if d.strip()]

INS = ""
with open("prompts/extscus.txt", "r") as f:
    INS = f.read().strip()

def save_txt(path:str, data:list[str]):
    if not os.path.exists("/".join(path.split("/")[:-2])):
        os.makedirs("/".join(path.split("/")[:-2]))

    with open(path, "w") as f:
        for item in data:
            f.write("%s\n" % item)
        f.close()
    return None

def add_txt(path:str, text:str):
    with open(path, "a") as f:
        f.write("%s\n" % text)
    return None

def get_summaries(model:str, is_val:bool = False):
    if is_val:
        with open(f"data/xsum_long/summ-val/{model}_resp", "r") as f:
            summaries = f.readlines()
    else:
        with open(f"data/xsum_long/summ/{model}_resp", "r") as f:
            summaries = f.readlines()

    return [s.strip() for s in summaries if s.strip()]

def get_batch_summaries(models:list[str], is_val:bool = False):
    batch_summaries = {}
    for model in models:
        batch_summaries[model] = get_summaries(model, is_val)
    return batch_summaries


def get_scus(summary:str, model:str):
    try:
        scus = call_llm(INS, summary, model)
        return scus
    except Exception as e:
        print(f"Error generating SCUs for model {model}: {e}")
        return None
   

def get_batch_scus(batch_summaries:dict[str, list[str]], model:str):
    batch_scus = {}
    summaries = []
    summaries_tag = []
    for m, s in batch_summaries.items():
        print(f"{m} - Summaries count: {len(s)}")
        summaries.extend(s)
        summaries_tag.extend([str(m)] * len(s))
    
    print(f"Total summaries count: {len(summaries)}")
    print(f"Sample tag: {summaries_tag[:5]}, {summaries_tag[100:105]}")

    scus_res = {}
    if summaries:
        summary_len = len(summaries)

        os.makedirs("tmp_3", exist_ok=True)
        os.makedirs("tmp_4", exist_ok=True)
        first_batch_scus = batch_call_llm(INS, summaries[:len(summaries)//2], model, "tmp_3", is_raw=True)
        sec_batch_scus = batch_call_llm(INS, summaries[len(summaries)//2:], model, "tmp_4", is_raw=True)
        batch_scus = first_batch_scus + sec_batch_scus
        for tag, scu in zip(summaries_tag, batch_scus):
            if tag not in scus_res:
                scus_res[tag] = []
            if ": " in scu:
                scu = " ".join(scu.split(": ")[1:]).strip()
            scus_res[tag].append(scu)
    else:
        print("No summaries found to generate SCUs.")
    
    return scus_res