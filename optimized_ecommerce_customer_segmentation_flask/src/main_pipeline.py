import os
import sqlite3

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import skfuzzy as fuzz

# Reference date for RFM recency (used in SQL script placeholder)
REFERENCE_DATE = "2025-11-11"


def get_paths():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "data")
    sql_dir = os.path.join(base_dir, "..", "sql")
    output_dir = os.path.join(base_dir, "..", "output")
    os.makedirs(output_dir, exist_ok=True)
    return data_dir, sql_dir, output_dir


def init_db(db_path, raw_csv_path):
    print(f"[DB] Initializing SQLite database at: {db_path}")
    df = pd.read_csv(raw_csv_path, parse_dates=["order_date"])

    conn = sqlite3.connect(db_path)

    customers_df = (
        df.groupby("customer_id")["order_date"]
        .min()
        .reset_index()
        .rename(columns={"order_date": "signup_date"})
    )

    customers_df.to_sql("customers", conn, if_exists="replace", index=False)
    df.to_sql("orders", conn, if_exists="replace", index=False)

    conn.close()
    print("[DB] Created tables: customers, orders")


def compute_rfm_sql(db_path, sql_path):
    print(f"[SQL] Computing RFM using script: {sql_path}")
    conn = sqlite3.connect(db_path)
    with open(sql_path, "r", encoding="utf-8") as f:
        sql_script = f.read()

    sql_script = sql_script.replace("{REFERENCE_DATE}", REFERENCE_DATE)
    conn.executescript(sql_script)
    conn.commit()
    conn.close()
    print("[SQL] RFM table created as rfm_table")


def load_rfm(db_path):
    conn = sqlite3.connect(db_path)
    df_rfm = pd.read_sql_query("SELECT * FROM rfm_table", conn)
    conn.close()
    print(f"[RFM] Loaded {len(df_rfm)} customer rows from rfm_table")
    return df_rfm


def preprocess_rfm(df_rfm):
    features = df_rfm[["recency_days", "frequency", "monetary"]].values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    print("[Preprocess] RFM features standardized")
    return X_scaled, scaler


def run_kmeans(df_rfm, X_scaled, output_dir, k_min=3, k_max=6, random_state=42):
    print("[KMeans] Searching best k using silhouette score")
    best_k = None
    best_score = -1.0
    best_model = None
    scores = []

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append((k, score))
        print(f"  k={k}, silhouette={score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
            best_model = km

    print(f"[KMeans] Best k = {best_k} (silhouette={best_score:.4f})")

    df_scores = pd.DataFrame(scores, columns=["k", "silhouette"])
    df_scores.to_csv(os.path.join(output_dir, "kmeans_k_selection.csv"), index=False)

    labels = best_model.labels_
    df_rfm["kmeans_cluster"] = labels
    df_rfm.to_csv(os.path.join(output_dir, "rfm_with_kmeans.csv"), index=False)

    summary = df_rfm.groupby("kmeans_cluster")[["recency_days", "frequency", "monetary"]].mean()
    summary.to_csv(os.path.join(output_dir, "kmeans_cluster_summary.csv"))

    plt.figure()
    plt.scatter(df_rfm["frequency"], df_rfm["monetary"], c=labels)
    plt.xlabel("Frequency")
    plt.ylabel("Monetary")
    plt.title("K-Means Clusters (Frequency vs Monetary)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "kmeans_clusters_frequency_monetary.png"))
    plt.close()

    return df_rfm, X_scaled, best_k, best_score


def run_fuzzy_cmeans(df_rfm, X_scaled, output_dir, n_clusters):
    print(f"[Fuzzy C-Means] Running with c={n_clusters} clusters")
    data = X_scaled.T

    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data=data,
        c=n_clusters,
        m=2.0,
        error=0.005,
        maxiter=1000,
        init=None,
    )

    cluster_labels = np.argmax(u, axis=0)

    for i in range(n_clusters):
        df_rfm[f"fuzzy_c{i}"] = u[i]
    df_rfm["fuzzy_cluster"] = cluster_labels
    df_rfm.to_csv(os.path.join(output_dir, "rfm_with_fuzzy_cmeans.csv"), index=False)

    centers_df = pd.DataFrame(cntr, columns=["recency_scaled", "frequency_scaled", "monetary_scaled"])
    centers_df.to_csv(os.path.join(output_dir, "fuzzy_cmeans_centers.csv"), index=False)

    with open(os.path.join(output_dir, "fuzzy_cmeans_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Fuzzy partition coefficient (FPC): {fpc:.4f}\n")

    print(f"[Fuzzy C-Means] Completed with FPC={fpc:.4f}")
    return df_rfm


def evaluate_weights(weights, X_scaled, n_clusters, random_state=42):
    weights = np.clip(weights, 0.0001, None)
    weights = weights / np.sum(weights)
    Xw = X_scaled * weights

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=5)
    labels = km.fit_predict(Xw)
    if len(set(labels)) < 2:
        return -1.0
    score = silhouette_score(Xw, labels)
    return score


def whale_optimization_feature_weights(X_scaled, n_clusters, output_dir,
                                       n_whales=15, max_iter=20, random_state=42):
    print("[Whale Optimization] Starting optimization of RFM feature weights")
    rng = np.random.default_rng(random_state)
    dim = X_scaled.shape[1]

    positions = rng.random((n_whales, dim))
    for i in range(n_whales):
        positions[i] = positions[i] / positions[i].sum()

    fitness = np.array([evaluate_weights(w, X_scaled, n_clusters, random_state) for w in positions])
    best_idx = np.argmax(fitness)
    best_pos = positions[best_idx].copy()
    best_fit = fitness[best_idx]

    history = []

    for t in range(max_iter):
        a = 2 - 2 * t / max(max_iter - 1, 1)
        for i in range(n_whales):
            r1 = rng.random()
            r2 = rng.random()
            A = 2 * a * r1 - a
            C = 2 * r2
            p = rng.random()

            if p < 0.5:
                if abs(A) < 1:
                    D = np.abs(C * best_pos - positions[i])
                    new_pos = best_pos - A * D
                else:
                    rand_idx = rng.integers(0, n_whales)
                    X_rand = positions[rand_idx]
                    D = np.abs(C * X_rand - positions[i])
                    new_pos = X_rand - A * D
            else:
                b = 1
                l = (rng.random() * 2) - 1
                D_prime = np.abs(best_pos - positions[i])
                new_pos = D_prime * np.exp(b * l) * np.cos(2 * np.pi * l) + best_pos

            new_pos = np.clip(new_pos, 0.0001, 1.0)
            new_pos = new_pos / new_pos.sum()
            positions[i] = new_pos

        fitness = np.array([evaluate_weights(w, X_scaled, n_clusters, random_state) for w in positions])
        iter_best_idx = np.argmax(fitness)
        iter_best_fit = fitness[iter_best_idx]

        if iter_best_fit > best_fit:
            best_fit = iter_best_fit
            best_pos = positions[iter_best_idx].copy()

        history.append((t + 1, best_fit))
        print(f"  Iteration {t + 1}/{max_iter}: best silhouette = {best_fit:.4f}")

    hist_df = pd.DataFrame(history, columns=["iteration", "best_silhouette"])
    hist_df.to_csv(os.path.join(output_dir, "whale_optimization_history.csv"), index=False)

    with open(os.path.join(output_dir, "whale_optimization_result.txt"), "w", encoding="utf-8") as f:
        f.write(f"Best feature weights [Recency, Frequency, Monetary]: {best_pos}\n")
        f.write(f"Best silhouette score: {best_fit:.4f}\n")

    print("[Whale Optimization] Completed")
    print(f"  Best weights: {best_pos}")
    print(f"  Best silhouette: {best_fit:.4f}")
    return best_pos, best_fit


def main():
    data_dir, sql_dir, output_dir = get_paths()
    raw_csv_path = os.path.join(data_dir, "raw_orders.csv")
    db_path = os.path.join(data_dir, "ecommerce.db")
    sql_path = os.path.join(sql_dir, "rfm_aggregation.sql")

    init_db(db_path, raw_csv_path)
    compute_rfm_sql(db_path, sql_path)

    df_rfm = load_rfm(db_path)
    X_scaled, _ = preprocess_rfm(df_rfm)

    df_rfm, X_scaled, best_k, base_sil = run_kmeans(df_rfm, X_scaled, output_dir)
    df_rfm = run_fuzzy_cmeans(df_rfm, X_scaled, output_dir, n_clusters=best_k)
    best_weights, best_sil = whale_optimization_feature_weights(X_scaled, best_k, output_dir)

    Xw = X_scaled * best_weights
    km_opt = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df_rfm["kmeans_whale_cluster"] = km_opt.fit_predict(Xw)
    df_rfm.to_csv(os.path.join(output_dir, "rfm_with_kmeans_and_whale.csv"), index=False)

    with open(os.path.join(output_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Best k: {best_k}\n")
        f.write(f"Baseline silhouette: {base_sil:.4f}\n")
        f.write(f"Whale-optimized silhouette: {best_sil:.4f}\n")

    print("\n[SUMMARY]")
    print(f"  Baseline K-Means silhouette (equal weights): {base_sil:.4f}")
    print(f"  Optimized K-Means silhouette (whale weights): {best_sil:.4f}")
    print("[DONE] Pipeline finished. Check the 'output' folder for results.")


if __name__ == "__main__":
    main()
