from utils import load_jsonl, save_jsonl

# since there are several issues in GPTRank, we need to test and repair the rank
def test_and_repair_rank(rank: list[int], num:int):
    exist_rank = set()
    repaired_rank = []
    for i, r in enumerate(rank):
        if i == num: break
        if r in exist_rank:
            repaired_rank.append(-1)
            continue
        repaired_rank.append(r)
        exist_rank.add(r)

    if len(repaired_rank) < num:
        print(f"Rank length {len(repaired_rank)} is less than expected {num}, repairing...")
        repaired_rank += [-1] * (num - len(repaired_rank))

    missing_rank = set(range(0, num)) - set(repaired_rank)
    for i in range(num):
        if repaired_rank[i] == -1:
            repaired_rank[i] = missing_rank.pop()
    return repaired_rank


def build_gptrank_train_data(data: list[dict], ranks: list[int]):
    # the rank is start from 1
    # the rank is a list of int, the length is the number of candidates
    assert len(data) == len(ranks), f"data length {len(data)} != ranks length {len(ranks)}"

    ranks = [test_and_repair_rank(rank, len(data[0]["candidates"])) for rank in ranks]
    train_data = []
    for raw_data, rank in zip(data, ranks):
        _candidates = raw_data["candidates"] if isinstance(raw_data["candidates"], list) else list(raw_data["candidates"].values())
        article = raw_data["article"]

        scores = [None] * len(_candidates)
        for i, r in enumerate(rank):
            scores[r-1] = len(rank) - i

        # the highest score candidate -> abstract
        abstract = _candidates[rank[0]-1]

        _candidates.pop(rank[0]-1)
        scores.pop(rank[0]-1)

        candidates = [(c, s) for c, s in zip(_candidates, scores)]

        train_data.append({
            "article": article,
            "abstract": abstract,
            "candidates": candidates
        })
    return train_data

def build_scurank_train_data(data: list[dict], ranks: list[int]):
    train_data = []
    for raw_data, rank in zip(data, ranks):
        _candidates = raw_data["candidates"] if isinstance(raw_data["candidates"], list) else list(raw_data["candidates"].values())
        article = raw_data["article"]

        scores = rank
        abstract = _candidates[rank.index(max(rank))]
        _candidates.pop(rank.index(max(rank)))
        scores.pop(rank.index(max(rank)))
        candidates = [(c, s) for c, s in zip(_candidates, scores)]
        train_data.append({
            "article": article,
            "abstract": abstract,
            "candidates": candidates
        })
    return train_data

def main(args):
    data_type = args.data_type
    rank_type = args.rank_type
    if rank_type == "scurank":
        cluster_type = args.cluster_type
        emb_type = args.emb_type
        for i in args.nums:
            train_rank = load_jsonl(f"rank_res/{data_type}/scurank/{cluster_type}/{emb_type}/train_ranks_{i}.jsonl")
            val_rank = load_jsonl(f"rank_res/{data_type}/scurank/{cluster_type}/{emb_type}/val_ranks_{i}.jsonl")
            train_candidates = load_jsonl(f"data/{data_type}/candidates.jsonl")
            val_candidates = load_jsonl(f"data/{data_type}/candidates-val.jsonl")
            train_data = build_scurank_train_data(train_candidates, train_rank)
            val_data = build_scurank_train_data(val_candidates, val_rank)
            save_jsonl(train_data, f"rank_res/{data_type}/scurank/{cluster_type}/{emb_type}/train-{i}.jsonl")
            save_jsonl(val_data, f"rank_res/{data_type}/scurank/{cluster_type}/{emb_type}/val-{i}.jsonl")
    elif rank_type == "gptrank":
        for i in args.nums:
            train_rank = load_jsonl(f"rank_res/{data_type}/gptrank/rank-{i}/train_ranks.jsonl")
            val_rank = load_jsonl(f"rank_res/{data_type}/gptrank/rank-{i}/val_ranks.jsonl")
            train_candidates = load_jsonl(f"data/{data_type}/candidates.jsonl")
            val_candidates = load_jsonl(f"data/{data_type}/candidates-val.jsonl")
            train_data = build_gptrank_train_data(train_candidates, train_rank)
            val_data = build_gptrank_train_data(val_candidates, val_rank)
            save_jsonl(train_data, f"rank_res/{data_type}/gptrank/rank-{i}/train.jsonl")
            save_jsonl(val_data, f"rank_res/{data_type}/gptrank/rank-{i}/val.jsonl")
    else:
        raise ValueError(f"Unknown rank type: {rank_type}")

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Build training data for GPTRank.")
    parser.add_argument("--data_type", type=str, required=True, help="Data type: 'train' or 'val'")
    parser.add_argument("--rank_type", type=str, required=True, choices=["gptrank", "scurank"], help="Rank type: 'gptrank' or 'scurank'")
    parser.add_argument("--cluster_type", type=str, default="kmeans", help="Cluster type for scurank")
    parser.add_argument("--emb_type", type=str, default="bert", help="Embedding type for scurank")
    parser.add_argument("--nums", nargs="+", type=int, default=[1], help="List of numbers for rank files")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)