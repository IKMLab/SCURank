import pandas as pd
import json, random, os

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



def main(args):
    if args.type == "cnn":
        data = load_jsonl("data/cnn_compare_file.jsonl")
    elif args.type == "xsum":
        data = load_jsonl("data/xsum_compare_file.jsonl")
    else:
        raise ValueError(f"Unknown type: {args.type}")
    
    batch = []
    for item in data[:30]:

        item["article"] = item["article"].replace("\n", " ").strip()
        item["gpt_summary"] = item["gpt_summary"].replace("\n", " ").strip()
        item["scu_summary"] = item["scu_summary"].replace("\n", " ").strip()

        rand = random.random() > 0.5
        batch.append({
            "article": item["article"],
            "summary1": item["scu_summary"] if rand else item["gpt_summary"],
            "summary2": item["gpt_summary"] if rand else item["scu_summary"],
            "label": "scu-gpt" if rand else "gpt-scu",
            "data_label": args.type
        })
    
    if os.path.exists(args.odir) is False:
        os.makedirs(args.odir)
    # turn to csv
    df = pd.DataFrame(batch)
    df.to_csv(f"{args.odir}/{args.type}_batch.csv", index=False)


def merge(cnn_pth: str, xsum_pth: str, out_pth: str):
    cnn_data = pd.read_csv(cnn_pth)
    xsum_data = pd.read_csv(xsum_pth)
    merged = pd.concat([cnn_data, xsum_data], ignore_index=True)
    merged.to_csv(out_pth, index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="cnn", help="Type of dataset to process")
    parser.add_argument("--odir", type=str, default="data/rand", help="Output file path")
    args = parser.parse_args()
    if args.type in ["merge"]:
        merge(
            "data/rand/cnn_batch.csv",
            "data/rand/xsum_batch.csv",
            "data/rand/merged_batch.csv"
        )
    else:
        main(args)