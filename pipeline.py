from utils import call_llm
from gen_scu import get_scus
from scurank import scurank

# Reference list: the models reported in the paper.
# Provided for reproducibility only — not used by main(). For a quick test,
# prefer the OpenAI-only `models` list below (faster setup, no GCP required).
#
# Swapping models:
#   - xAI / Anthropic / OpenAI / Gemini families work out of the box.
#   - Other Llama or Mistral variants require changes in util/gcloud.py.
models_sample = [
    "gemini-1.5-flash",  # gemini 1.5 versions are deprecated
    "gemini-1.5-pro",
    "meta/llama-3.1-70b-instruct-maas",
    "meta/llama-3.1-405b-instruct-maas",
    "mistral-large-2411",
    "claude-3-5-sonnet-20240620",
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
]

models = [
    "gpt-4o-mini",
    "gpt-5-nano",
    "gpt-5-mini",
    "gpt-5.4-nano",
]

sample = "Floyd Mayweather Jnr continued his preparations for his fight with Manny Pacquiao in May\u00a0on day two of his official training camp in Las Vegas. The undefeated welterweight champion kept fans updated of his progress with a series of pictures uploaded to social media website Shots. Mayweather, who currently holds the WBC, WBA and Ring titles, shared snaps from within the gym as he was put through a rigorous pad workout by his trainer. Floyd Mayweather is put through his paces on day two of his official training camp in Las Vegas. The undefeated welterweight champion goes through a pad workout with his uncle, Roger Mayweather. Mayweather is preparing for his May 2 fight with Manny Pacquiao at the MGM Grand. Mayweather is given a pep talk by his father Floyd Snr during the second day of training on Tuesday. Mayweather and Pacquiao finally agreed terms on the fight after protracted negotiations. Mayweather will work with his father and uncle and has, according to reports, already booked the services of former opponent DeMarcus Corley as a sparring partner. The $300million bout was finally agreed, after much posturing from both camps, for May 2 at the MGM Grand in Vegas and both fighters have begun their quest for peak physical condition. The pair will come face-to-face next week for the only time before the week of the fight when they hold a press conference in Los Angeles on March 11. Pacquiao takes to the ring in his Los Angeles gym to show off his speed of movemet. Pacquiao is upping the intensity of his training with just two months to go before he faces Mayweather. Former member of The Money Team rapper 50 Cent revealed on Tuesday that he was planning to put a $1m bet on Mayweather winning the fight. Pacquiao started his training camp on the same day as Mayweather and has also been active on social media, posting updates on his preparations. The Filipino star began his regime in trainer Freddie Roach's Wild Card Gym in LA, which will be completely shut down next week to preserve the secrecy of his game plan."

def main():
    # First Step: Summary Generation
    summ_ins = ""
    with open("prompts/summ_cnn.txt", "r") as f:
        summ_ins = f.read()

    summaries = [call_llm(summ_ins, sample, model) for model in models]
    for i, model in enumerate(models):
        print(f"Model: {model}\nSummary: {summaries[i]}\n")

    # Second Step: SCU Generation
    model = "gpt-4o-mini"
    scus = [get_scus(summ, model) for summ in summaries]
    scus = [scu.split(" # ") for scu in scus]
    for i, model in enumerate(models):
        print(f"Model: {model}\nSCUs: {scus[i]}\n")

    # Third Step: SCURank
    data = {"article": sample, "candidates": summaries, "scus": scus} # Structure for scurank_raw
    rank = scurank([data], cluster_type="hdbscan", emb_type="all-mpnet-base-v2")
    print(f"SCURank Results: {rank[0]}\n")
    sorted_models = sorted(zip(models, rank[0]), key=lambda x: x[1])
    print("Ranking of Models (Best to Worst):")
    for model, rank in sorted_models:
        print(f"{rank}: {model}")

    return rank

if __name__ == "__main__":
    main()