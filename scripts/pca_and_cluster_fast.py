
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, json, time, warnings
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, DBSCAN, Birch
from sklearn.mixture import GaussianMixture
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def safe_internal_metrics(X, labels):
    k = len(np.unique(labels))
    n = X.shape[0]
    if k < 2 or k >= n:
        return np.nan, np.nan, np.nan
    try: sil = silhouette_score(X, labels, metric="euclidean")
    except Exception: sil = np.nan
    try: ch = calinski_harabasz_score(X, labels)
    except Exception: ch = np.nan
    try: db = davies_bouldin_score(X, labels)
    except Exception: db = np.nan
    return sil, ch, db

def hungarian_best_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    from scipy.optimize import linear_sum_assignment
    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)
    if labels_true.size == 0 or labels_pred.size == 0: return float("nan")
    matrix = np.zeros((labels_pred.size, labels_true.size), dtype=int)
    for i, lp in enumerate(labels_pred):
        for j, lt in enumerate(labels_true):
            matrix[i, j] = np.sum((y_pred == lp) & (y_true == lt))
    max_val = matrix.max() if matrix.size else 0
    cost = max_val - matrix
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {labels_pred[r]: labels_true[c] for r, c in zip(row_ind, col_ind)}
    y_mapped = np.array([mapping.get(p, labels_true[0]) for p in y_pred])
    return f1_score(y_true, y_mapped, average="macro")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--n-components", type=int, default=150)
    ap.add_argument("--k-range", type=str, default="2:10")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("[1/5] Loading data")
    X = sparse.load_npz(args.features)
    with open(args.labels, "r", encoding="utf-8") as f:
        lbl = json.load(f)
    y_labels = np.array(lbl["labels"])
    uniq = {l:i for i,l in enumerate(sorted(set(y_labels)))}
    y_true = np.array([uniq[l] for l in y_labels], dtype=int)

    print("[2/5] TruncatedSVD")
    svd = TruncatedSVD(n_components=min(args.n_components, min(X.shape)-1), random_state=args.random_state)
    X_red = svd.fit_transform(X)
    scaler = StandardScaler(with_mean=True)
    X_scaled = scaler.fit_transform(X_red)

    print("[3/5] Building jobs")
    a,b = [int(x) for x in args.k_range.split(":")]
    ks = list(range(a, b+1))
    jobs = []
    for k in ks:
        jobs += [("KMeans", {"k": k}), ("MiniBatchKMeans", {"k": k, "batch_size": 512}),
                 ("Agglomerative(ward)", {"k": k}), ("GaussianMixture", {"k": k, "covariance_type": "full"}),
                 ("Birch", {"k": k, "threshold": 0.5, "branching_factor": 50})]
    for eps in [0.8, 1.2]:
        for ms in [5, 10]:
            jobs.append(("DBSCAN", {"eps": eps, "min_samples": ms}))

    print(f"[4/5] Running {len(jobs)} jobs")
    results = []
    done = 0; total = len(jobs)
    for name, params in jobs:
        start = time.time()
        try:
            if name == "KMeans":
                model = KMeans(n_clusters=params["k"], n_init="auto", random_state=args.random_state)
                pred = model.fit_predict(X_scaled)
            elif name == "MiniBatchKMeans":
                model = MiniBatchKMeans(n_clusters=params["k"], random_state=args.random_state, batch_size=params["batch_size"])
                pred = model.fit_predict(X_scaled)
            elif name == "Agglomerative(ward)":
                model = AgglomerativeClustering(n_clusters=params["k"], linkage="ward")
                pred = model.fit_predict(X_scaled)
            elif name == "GaussianMixture":
                model = GaussianMixture(n_components=params["k"], covariance_type=params["covariance_type"], random_state=args.random_state)
                pred = model.fit_predict(X_scaled)
            elif name == "Birch":
                model = Birch(n_clusters=params["k"], threshold=params["threshold"], branching_factor=params["branching_factor"])
                pred = model.fit_predict(X_scaled)
            elif name == "DBSCAN":
                model = DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
                pred = model.fit_predict(X_scaled)
            else:
                raise ValueError("Unknown algo")
            sil, ch, db = safe_internal_metrics(X_scaled, pred)
            f1 = hungarian_best_f1(y_true, pred)
            elapsed = time.time() - start
            results.append({"algo":name,"params":params,"silhouette":sil,"calinski_harabasz":ch,
                            "davies_bouldin":db,"f1_macro_mapped":f1,"elapsed_sec":elapsed,
                            "n_clusters":int(len(np.unique(pred)))})
        except Exception as e:
            results.append({"algo":name,"params":params,"silhouette":np.nan,"calinski_harabasz":np.nan,
                            "davies_bouldin":np.nan,"f1_macro_mapped":np.nan,"elapsed_sec":time.time()-start,
                            "n_clusters":-1,"error":str(e)})
        done += 1
        bar = int((done/total)*30)
        print(f"  [{done:3d}/{total}] [{'#'*bar}{'.'*(30-bar)}] {name} {params}")

    df = pd.DataFrame(results)
    df["params_json"] = df["params"].apply(lambda d: json.dumps(d, sort_keys=True))
    out_csv = os.path.join(args.out_dir, "metrics.csv")
    df.to_csv(out_csv, index=False)

    def normalize(col, high=True):
        c = df[col].astype(float)
        if col == "davies_bouldin": high=False
        if c.notna().sum() < 2:
            import pandas as pd
            return pd.Series([0.0]*len(c))
        vals = (c - c.min()) / (c.max() - c.min() + 1e-12)
        if not high: vals = 1 - vals
        return vals

    score = 0.5*normalize("silhouette", True) + 0.3*normalize("calinski_harabasz", True) + 0.2*normalize("davies_bouldin", False)
    df["internal_score_norm"] = score

    best_idx = df["internal_score_norm"].idxmax()
    best_row = df.iloc[int(best_idx)].to_dict()
    best = {
        "best_algo": best_row["algo"],
        "best_params": json.loads(best_row["params_json"]),
        "internal_metrics": {
            "silhouette": float(best_row["silhouette"]) if pd.notna(best_row["silhouette"]) else None,
            "calinski_harabasz": float(best_row["calinski_harabasz"]) if pd.notna(best_row["calinski_harabasz"]) else None,
            "davies_bouldin": float(best_row["davies_bouldin"]) if pd.notna(best_row["davies_bouldin"]) else None,
            "internal_score_norm": float(best_row["internal_score_norm"]),
        },
        "external_f1_macro_mapped": float(best_row["f1_macro_mapped"]) if pd.notna(best_row["f1_macro_mapped"]) else None,
        "n_clusters": int(best_row.get("n_clusters", -1)),
    }
    with open(os.path.join(args.out_dir, "best_config.json"), "w") as f:
        json.dump(best, f, indent=2)

    plots = os.path.join(args.out_dir, "plots"); os.makedirs(plots, exist_ok=True)
    plt.figure(); df.groupby("algo")["internal_score_norm"].max().sort_values(ascending=False).plot(kind="bar")
    plt.ylabel("Best normalized internal score"); plt.tight_layout(); plt.savefig(os.path.join(plots, "best_internal_score_by_algo.png")); plt.close()
    plt.figure(); df.groupby("algo")["f1_macro_mapped"].max().sort_values(ascending=False).plot(kind="bar")
    plt.ylabel("Best external F1 (Hungarian)"); plt.tight_layout(); plt.savefig(os.path.join(plots, "best_f1_by_algo.png")); plt.close()

    # 2D scatter (componentes 1 e 2) com clusters do best
    name = best["best_algo"]; params = best["best_params"]
    if name == "KMeans":
        pred = KMeans(n_clusters=params["k"], n_init="auto", random_state=args.random_state).fit_predict(X_scaled)
    elif name == "MiniBatchKMeans":
        pred = MiniBatchKMeans(n_clusters=params["k"], random_state=args.random_state, batch_size=params.get("batch_size",512)).fit_predict(X_scaled)
    elif name == "Agglomerative(ward)":
        pred = AgglomerativeClustering(n_clusters=params["k"], linkage="ward").fit_predict(X_scaled)
    elif name == "GaussianMixture":
        pred = GaussianMixture(n_components=params["k"], covariance_type=params.get("covariance_type","full"), random_state=args.random_state).fit_predict(X_scaled)
    elif name == "Birch":
        pred = Birch(n_clusters=params.get("k", None), threshold=params.get("threshold",0.5), branching_factor=params.get("branching_factor",50)).fit_predict(X_scaled)
    elif name == "DBSCAN":
        pred = DBSCAN(eps=params["eps"], min_samples=params["min_samples"]).fit_predict(X_scaled)
    else:
        pred = np.zeros(X_scaled.shape[0], dtype=int)

    plt.figure()
    plt.scatter(X_scaled[:,0], X_scaled[:,1], s=8, alpha=0.6, c=pred)
    plt.xlabel("Component 1"); plt.ylabel("Component 2"); plt.title(f"Clusters - {name}")
    plt.tight_layout(); plt.savefig(os.path.join(plots, "scatter_best_clusters_2D.png")); plt.close()

    print("[5/5] Done")
    print("CSV:", out_csv)
    print("BEST:", os.path.join(args.out_dir, "best_config.json"))
    print("PLOTS:", plots)

if __name__ == "__main__":
    main()
