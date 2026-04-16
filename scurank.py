import numpy as np
from tqdm import tqdm
import torch
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, AffinityPropagation
import warnings, os, umap, argparse
from utils import *

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

"""
default data type: list[dict{"article", "candidates": {dict[str]}, "scus": {dict[list[str]]}}]
"""
def generate_scus(candidates):
    print("Generating SCUs...")
    if isinstance(candidates, dict):
        candidates = list(candidates.values())
    scus = []
    instruction = ""
    with open("prompts/prompt_scus.txt", "r") as f:
        instructions = f.read()
    for candidate in candidates:
        scus.append(call_llm(instructions, candidate, "gpt-4o-mini"))
    return scus

def _embedding(scus: list[list[str]], model: SentenceTransformer) -> tuple[list[int], list[float]]:
    """
    embedded: return ids and points
    """
    # scus = [[preprocess_text(s) for s in _scu] for _scu in scus]
    encoded = [model.encode(s) for s in scus]
    ids = [i  for i, s in enumerate(scus) for _ in range(len(s))]
    points = [e for enc in encoded for e in enc]

    return ids, points

def clusterer(cluster_type) -> object:
    if cluster_type == "hdbscan-umap":
        return HDBSCAN(min_cluster_size=3, 
                      min_samples=2,
                      metric='correlation', 
                      n_jobs=8, 
                      allow_single_cluster=True,
                      cluster_selection_epsilon=0.004, 
                    )
    elif cluster_type == "hdbscan":
        return HDBSCAN(min_cluster_size=2,
                       min_samples=2,
                       metric='correlation',
                       n_jobs=8,
                       allow_single_cluster=True,
                       cluster_selection_epsilon=0.15,
                       )
    elif cluster_type == "agglomerative":
        return AgglomerativeClustering(n_clusters=None, distance_threshold=0.9, linkage='ward')
    elif cluster_type == "affinity":
        return AffinityPropagation(damping=0.7, preference=-20)
    else:
        raise ValueError(f"Unknown cluster type: {cluster_type}")

def _score(labels: list[int], ids: list[int], pl: list[float]) -> list[float]:
    scores = [0] * (max(ids) + 1)
    for i, l in zip(ids, labels):
        if l == -1: scores[i] += 1
        else:
            scores[i] += labels.count(l)

    scores = np.array(scores) / np.array(pl)
    return scores
import string
def __word_count(text: str) -> int:
    # Slowest but most accurate
    tokens = word_tokenize(text)
    words = [word for word in tokens if word not in string.punctuation]
    # Moderate Method
    # words = re.findall(r'\b\w+\b', text)
    # Fastest but least accurate
    # words = text.split()
    return len(words)

def _penalty(candidates: list[str]) -> list[float]:
    """
    penalty: sqrt(len(summary))
    """
    return np.sqrt([__word_count(s) for s in candidates]).tolist()

def scurank_raw(raw_data, cluster, emb_model, is_umap:bool=False, corrected_umap:umap.UMAP=None) -> list[int]:
    ### check raw_data format
    if not isinstance(raw_data["candidates"], list):
        raw_data["candidates"] = list(raw_data["candidates"].values())
    if not isinstance(raw_data["scus"], list):
        raw_data["scus"] = list(raw_data["scus"].values())

    # umap
    if (is_umap == False) and (corrected_umap is not None):
        print("Warning: is_umap is False but corrected_umap is provided. Using corrected_umap.")
        is_umap = True
    if corrected_umap is not None:
        reducer = corrected_umap
    else: 
        reducer = umap.UMAP(n_neighbors=5, n_components=15, min_dist=0.2, metric='cosine')
    
    score = [0] * len(raw_data["candidates"])
    pl = _penalty(raw_data["candidates"])
    ids, points = _embedding(raw_data["scus"], emb_model)
    
    if is_umap:
        points = reducer.fit_transform(points)
    labels = cluster.fit(points).labels_.tolist()
    score = _score(labels, ids, pl)
    rank = (np.argsort(score)[::-1] + 1).tolist()
    return rank

def scurank(data, cluster_type, emb_type = "all-mpnet-base-v2", is_generate_scus:bool = False, is_umap:bool = False, corrected_umap:umap.UMAP = None, corrected_cluster = None) -> list[list[int]]:
    """
    SCURank: Sentence Clustering for Unsupervised Ranking
    Args:
        data: list of raw data
        cluster_type: type of clustering algorithm (hdbscan, agglomerative, affinity)
        emb_type: type of embedding model
        is_generate_scus: whether to generate SCUs using OpenAI API
        is_umap: whether to use UMAP for dimensionality reduction
        corrected_umap: use own UMAP model (recommend to use when using the embedding model not in paper)
    """
    print("Loading embedding model...")
    emb_model = SentenceTransformer(emb_type, trust_remote_code=True, device="cuda" if torch.cuda.is_available() else "cpu")
    if corrected_cluster is None:
        cluster = clusterer(cluster_type)
    else:
        cluster = corrected_cluster

    if is_generate_scus:
        scus = []
    if not is_generate_scus and "scus" not in data[0]:
        raise ValueError("SCUs are not generated and not found in the data. Please generate SCUs first or set is_generate_scus to True.")
    ranks = []
    for raw_data in tqdm(data, mininterval=1, desc="Ranking"):
        if is_generate_scus:
            raw_data["scus"] = generate_scus(raw_data["candidates"])
            scus.append(raw_data["scus"])
        rank = scurank_raw(raw_data, cluster, emb_model, is_umap, corrected_umap)
        ranks.append(rank)

    if is_generate_scus: save_jsonl("scus.jsonl", scus)


    # del emb_model
    # gc.collect()
    # torch.cuda.empty_cache()
    return ranks


def main(args):
    train_data = load_jsonl(f"data/{args.type}/candidates.jsonl")
    val_data = load_jsonl(f"data/{args.type}/candidates-val.jsonl")

    train_ranks = scurank(train_data, args.cluster, args.emb, is_generate_scus=args.is_generate_scus)
    val_ranks = scurank(val_data, args.cluster, args.emb, is_generate_scus=args.is_generate_scus)

    # save ranks
    dir = f"rank_res/{args.type}/scurank/{args.cluster}/{args.emb}"
    if not os.path.exists(dir):
        os.makedirs(dir)

    if args.num != -1:
        train_save_pth = f"{dir}/train_ranks_{args.num}.jsonl"
        val_save_pth = f"{dir}/val_ranks_{args.num}.jsonl"
    else:
        train_save_pth = f"{dir}/train_ranks.jsonl"
        val_save_pth = f"{dir}/val_ranks.jsonl"
    
    save_jsonl(train_ranks, train_save_pth)
    save_jsonl(val_ranks, val_save_pth)

    print(f"train ranks saved to {train_save_pth}")
    print(f"val ranks saved to {val_save_pth}")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="cnn_llms")
    parser.add_argument("--scu_type", type=str, default="gpt")
    parser.add_argument("--cluster", type=str, default="hdbscan")
    parser.add_argument("--emb", type=str, default="all-mpnet-base-v2")
    parser.add_argument("--is_generate_scus", action="store_true")
    parser.add_argument("--is_umap", action="store_true")
    parser.add_argument("--num", type=int, default=-1)
    args = parser.parse_args()
    main(args)
    


def print_cluster(labels: list[int], ids: list[int], scus: list[list[str]]) -> None:
    """
    print all scus in the same cluster
    """
    store_text = ""
    labels = list(labels)
    scus = [ss for s in scus for ss in s]
    for i in range(len(set(labels))):
        store_text += f"Cluster {i}\n"
        print(f"Cluster {i}")
        for j in range(len(labels)):
            if labels[j] == i:
                store_text += f"Summary {ids[j]}: {scus[j]}\n"
                print(f"Summary {ids[j]}: {scus[j]}")
        store_text += "-"*20 + "\n"
        print("=====================================")
    if -1 in labels:
        store_text += f"Cluster -1\n"
        print(f"Cluster -1")
        for j in range(len(labels)):
            if labels[j] == -1:
                store_text += f"Summary {ids[j]}: {scus[j]}\n"
                print(f"Summary {ids[j]}: {scus[j]}")
        store_text += "-"*20 + "\n"
        print("=====================================")
    with open("cluster.txt", "w") as f:
        f.write(store_text)
    return None