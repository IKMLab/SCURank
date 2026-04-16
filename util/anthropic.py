# grok-2, claude-3.5-sonnet-v2, claude-3.5-haiu, mistral-large-2411, llama-3.2-90b, gemini-2.0-flash
import os, sys, time
from secret import GROK_API_KEY, ANTHROPIC_API_KEY
from anthropic import Anthropic

def build_grok_api():
    return Anthropic(api_key=GROK_API_KEY, base_url="https://api.x.ai")

def call_grok(ins:str, content:str, model:str, times:int=0):
    client = build_grok_api()
    if times > 10: raise ValueError("Exceeded the maximum number of retries")
    try:
        message = client.messages.create(
            model="grok-2-1212",
            messages=[
                {"role": "user", "content": ins},
                {"role": "user", "content": content},
            ],
            max_tokens=1024,
            temperature=0.0,
        )
    except Exception as e:
        print(e)
        time.sleep(10)
        return call_grok(ins, content, model, times+1)
    return message.content[0].text

def batch_call_grok(instruction:str, content:list, model:str, path:str):
    results = []
    for c in content:
        res = call_grok(instruction, c, model)
        results.append(res)
        if res.find("\n\n") != -1:
            res = " ".join(res.split("\n\n")[1:])
        if res.find("\n") != -1:
            res = " ".join(res.split("\n"))
        res = res.strip()
        print(res)
        with open(f"{path}/summary.txt", "a") as f:
            f.write(res + "\n")
    return results

# ------------- #

from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

def build_claude_api():
    return Anthropic(api_key=ANTHROPIC_API_KEY)

def call_claude(ins:str, content:str, model:str):
    client = build_claude_api()
    resp = client.messages.create(
        model=model,
        messages=[
            {"role": "user", "content": ins},
            {"role": "user", "content": content},
        ],
        max_tokens=512,
        temperature=0.0,
    )
    return resp.content[0].text

def batch_call_claude(ins:str, content:list, model:str):
    client = build_claude_api()
    message_batch = client.messages.batches.create(
        requests=[
            Request(
                custom_id=f"request-{i}",
                params=MessageCreateParamsNonStreaming(
                    model=model,
                    max_tokens=2048,
                    messages=[
                        {"role": "user", "content": ins + "\n\n" + c},
                    ]
                )
            )
            for i, c in enumerate(content)
        ]
    )
    id = message_batch.id
    while(True):
        time.sleep(10)
        message_batch = client.messages.batches.retrieve(id)
        print(f"Batch {message_batch.id} processing status is {message_batch.processing_status}", end='\r')
        if message_batch.processing_status == "succeeded" or message_batch.processing_status == "ended":
            break
    results = client.messages.batches.results(id)
    results = sorted(results, key=lambda x: int(x.custom_id.split("-")[-1]))
    results = [r.result.message.content[0].text for r in results]
    proceed_results = []
    for i, r in enumerate(results):
        if r.find("\n\n") != -1:
        #    print(r.split("\n\n"))
           r = " ".join(r.split("\n\n")[1:])
        #    print(results[i])
        if r.find("\n") != -1:
            results[i] = " ".join(r.split("\n"))
        r = r.strip()
        proceed_results.append(r)
    return proceed_results