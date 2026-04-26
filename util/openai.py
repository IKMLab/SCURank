from secret import OPENAI_API_KEY
from openai import OpenAI
import time, sys, json
def save_jsonl(data, pth: str):
    """
    Save a list of dictionaries to a JSONL file.
    """
    with open(pth, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def build_openai_api():
    return OpenAI(api_key=OPENAI_API_KEY)

def openai_batch_api_call(dir: str):
    client = build_openai_api()
    batch_input_file = client.files.create(
        file = open(f"{dir}/batch_input.jsonl", "rb"),
        purpose="batch"
    )
    batch_input_file_id = batch_input_file.id
    batches = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window='24h',
    )
    id = batches.id

    start_time = time.time()
    while(client.batches.retrieve(id).status != "completed"):
        time.sleep(1)
        sys.stdout.write(f"{client.batches.retrieve(id).status}...{int(time.time() - start_time)}s\r")
        sys.stdout.flush()
    sys.stdout.write("\n")
    
    response_id = client.batches.retrieve(id).output_file_id
    # res = client.files.content(response_id).content

    with open(f"{dir}/batch_output.jsonl", "wb") as f:
        f.write(client.files.content(response_id).content)
        f.close()

    return None

def create_batch_input_file(contents: list[str], dir: str, instruction: str = None, model_name: str="gpt-4o-mini") -> None:
    batch_input = []
    for i, content in enumerate(contents):
        if instruction is not None:
            body = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": content},
                ],
                "max_tokens": 512,
                "temperature": 0.0,
            }
        else:
            body = {
                "model": model_name,
                "messages": [
                    {"role": "user", "content": content},
                ],
                "max_tokens": 512,
                "temperature": 0.0,
            }
        custom_id = "requests_" + str(i)
        batch_input.append({
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body
        })
    save_jsonl(batch_input, f"{dir}/batch_input.jsonl")
    return None

def batch_call_openai(ins:str, contents:list[str], model:str, path:str) -> None:
    if ins == "" or None: ins = None
    create_batch_input_file(contents, path, ins, model)
    openai_batch_api_call(path)
    with open(f"{path}/batch_output.jsonl", "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    f.close()
    resps = [d["response"]["body"]["choices"][0]["message"]["content"] for d in data]
    return resps


def call_openai(ins:str = None, content:str = None, model:str="gpt-4o-mini",):
    # gpt "5" series models do not support `max_tokens` and `temperature` parameters, so we set them in the prompt instead.
    client = build_openai_api()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": content},
        ] if ins is None else [
            {"role": "system", "content": ins},
            {"role": "user", "content": content},
        ],
        # max_tokens=512,
        # temperature=0.0,
    )
    return response.choices[0].message.content