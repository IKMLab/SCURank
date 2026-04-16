import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from utils import call_llm, load_jsonl, load_txt, save_txt
from tqdm import tqdm
import re
llm_compare_prompt = """
Compare the following summaries based on **information richness** (how much relevant detail is preserved) and **importance** (focus on the most significant points).

For each summary, evaluate:
- Information richness: Comprehensiveness, specific details, coverage of key topics
- Importance: Focus on high-impact information, critical insights, actionable content

Provide a brief analysis and rank the summaries from best to worst, explaining your reasoning.

Input format:
Article: [article text]
Summary 1: [summary 1 text]
Summary 2: [summary 2 text]

Output format:
Summary 1 Analysis: [Your analysis here]
Summary 2 Analysis: [Your analysis here]
Winner: [1 or 2 or "tie" if they are equally good]
"""

def _load_cnn_distilled_resp(num:int, metric:str) -> list[str]:
    resp = load_jsonl(f"experiments/distilled_model_output/cnn_llms/{metric}_llms_{num}_result.jsonl")
    summaries = [r["summary"] for r in resp]
    articles = [r["article"] for r in resp]
    return articles, summaries

def _load_xsum_distilled_resp(num:int, metric:str) -> list[str]:
    resp = load_jsonl(f"experiments/distilled_model_output/xsum/{metric}_{num}_result.jsonl")
    summaries = [r["summary"] for r in resp]
    articles = [r["article"] for r in resp]
    return articles, summaries

def __extract_winner(resp:str) -> int:
    # Winner: [1 or 2 or "tie"]
    match = re.search(r'Winner.*?:.*?(\d+|tie|Tie)', resp)
    if not match:
        match = re.search(r'winner.*?:.*?(\d+|tie|Tie)', resp)
    if match:
        winner = match.group(1)
        if winner == "tie" or winner == "Tie":
            return 0
        return int(winner)
    print(f"Error: Could not extract winner from response: {resp}")
    return None

def __determine_winner(res1:int, res2:int)->int:
    if res2 == 1: res2 = 2
    elif res2 == 2: res2 = 1
    else: res2 = 0

    if res1 == res2 and res1 != 0: return res1
    elif res1 != res2 and res1 != 0 and res2 != 0: return 0
    elif res1 == 0 and res2 != 0: return res2
    elif res2 == 0 and res1 != 0: return res1
    else: return 0

def summ_compare(article:str, summ1:str, summ2:str, mname:str)->list[int]:
    
    prompt = f"Article: {article}\nSummary 1: {summ1}\nSummary 2: {summ2}\n"
    response = call_llm(ins=llm_compare_prompt, content=prompt, model=mname)
    # print(response)
    res1 = __extract_winner(response)
    
    prompt = f"Article: {article}\nSummary 1: {summ2}\nSummary 2: {summ1}\n"
    response = call_llm(ins=llm_compare_prompt, content=prompt, model=mname)
    # print(response)
    res2 = __extract_winner(response)
    winner = __determine_winner(res1, res2)

    return winner

def main_cnn(test_model:str="gpt-4.2"):
    art1, summ1 = _load_cnn_distilled_resp(3, "scurank")
    art2, summ2 = _load_cnn_distilled_resp(2, "gptrank")
    assert len(art1) == len(summ1) == len(art2) == len(summ2), "Data lengths do not match."
    for i in range(len(art1)):
        assert art1[i] == art2[i], f"Articles do not match at index {i}."
    
    res = []
    for i in tqdm(range(len(summ1))):
        # claude-sonnet-4-20250514
        winner = summ_compare(art1[i], summ1[i], summ2[i], test_model)
        # print(f"Article {i+1}: Winner is Summary {winner}")
        res.append(winner)
        save_txt(f"experiments/llms-compare/cnn_compare_gptrank_scurank_by_{test_model}.txt", res)
    print(f"1: {res.count(1)}, 2: {res.count(2)}, tie: {res.count(0)}")
    print(f"Results saved to experiments/llms-compare/cnn_compare_gptrank_scurank_by_{test_model}.txt")

def main_xsum(test_model:str="gpt-5.1"):
    art1, summ1 = _load_xsum_distilled_resp(10, "scurank")
    art2, summ2 = _load_xsum_distilled_resp(10, "gptrank")
    assert len(art1) == len(summ1) == len(art2) == len(summ2), "Data lengths do not match."
    for i in range(len(art1)):
        assert art1[i] == art2[i], f"Articles do not match at index {i}."
    
    res = []
    for i in tqdm(range(len(summ1))):
        winner = summ_compare(art1[i], summ1[i], summ2[i], test_model)
        # print(f"Article {i+1}: Winner is Summary {winner}")
        res.append(winner)
        save_txt(f"experiments/llms-compare/xsum_compare_gptrank_scurank_by_{test_model}.txt", res)
    print(f"1: {res.count(1)}, 2: {res.count(2)}, tie: {res.count(0)}")
    print(f"Results saved to experiments/llms-compare/xsum_compare_gptrank_scurank_by_{test_model}.txt")

if __name__ == "__main__":
    test_model = "gemini-2.5-pro"
    main_cnn(test_model)