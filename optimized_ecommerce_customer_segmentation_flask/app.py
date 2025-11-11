import os
from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
from sklearn.cluster import KMeans

from src.main_pipeline import (
    get_paths,
    init_db,
    compute_rfm_sql,
    load_rfm,
    preprocess_rfm,
    run_kmeans,
    run_fuzzy_cmeans,
    whale_optimization_feature_weights,
)

app = Flask(__name__)
app.secret_key = "change_this_for_production"


def run_full_pipeline(raw_csv_path):
    data_dir, sql_dir, output_dir = get_paths()
    os.makedirs(output_dir, exist_ok=True)
    db_path = os.path.join(data_dir, "ecommerce_web.db")
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

    summary_path = os.path.join(output_dir, "web_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Best k: {best_k}\n")
        f.write(f"Baseline K-Means silhouette: {base_sil:.4f}\n")
        f.write(f"Optimized (Whale) silhouette: {best_sil:.4f}\n")

    return best_k, base_sil, best_sil


@app.route("/", methods=["GET"])
def index():
    _, _, output_dir = get_paths()
    summary = None
    summary_file = os.path.join(output_dir, "web_summary.txt")
    if os.path.exists(summary_file):
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = f.read()
    return render_template("index.html", summary=summary)


@app.route("/run", methods=["POST"])
def run_pipeline():
    if "dataset" not in request.files:
        flash("No file part in the request.")
        return redirect(url_for("index"))

    file = request.files["dataset"]
    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("index"))

    data_dir, _, _ = get_paths()
    os.makedirs(data_dir, exist_ok=True)
    upload_path = os.path.join(data_dir, "web_orders.csv")
    file.save(upload_path)

    try:
        pd.read_csv(upload_path)
    except Exception as e:
        flash(f"Failed to read CSV: {e}")
        return redirect(url_for("index"))

    best_k, base_sil, best_sil = run_full_pipeline(upload_path)
    flash(
        f"Pipeline finished. Best k = {best_k}, baseline silhouette = {base_sil:.4f}, "
        f"optimized silhouette = {best_sil:.4f}."
    )
    return redirect(url_for("results"))


@app.route("/results", methods=["GET"])
def results():
    _, _, output_dir = get_paths()

    def safe_read_csv(name, nrows=None):
        path = os.path.join(output_dir, name)
        if not os.path.exists(path):
            return None
        try:
            return pd.read_csv(path, nrows=nrows)
        except Exception:
            return None

    kmeans_summary = safe_read_csv("kmeans_cluster_summary.csv")
    kmeans_k_sel = safe_read_csv("kmeans_k_selection.csv")
    rfm_kmeans = safe_read_csv("rfm_with_kmeans.csv", nrows=50)
    rfm_kmeans_whale = safe_read_csv("rfm_with_kmeans_and_whale.csv", nrows=50)
    fuzzy_centers = safe_read_csv("fuzzy_cmeans_centers.csv")
    fuzzy_metrics = None
    whale_result = None

    fuzzy_metrics_path = os.path.join(output_dir, "fuzzy_cmeans_metrics.txt")
    if os.path.exists(fuzzy_metrics_path):
        with open(fuzzy_metrics_path, "r", encoding="utf-8") as f:
            fuzzy_metrics = f.read()

    whale_result_path = os.path.join(output_dir, "whale_optimization_result.txt")
    if os.path.exists(whale_result_path):
        with open(whale_result_path, "r", encoding="utf-8") as f:
            whale_result = f.read()

    return render_template(
        "results.html",
        kmeans_summary=kmeans_summary,
        kmeans_k_sel=kmeans_k_sel,
        rfm_kmeans=rfm_kmeans,
        rfm_kmeans_whale=rfm_kmeans_whale,
        fuzzy_centers=fuzzy_centers,
        fuzzy_metrics=fuzzy_metrics,
        whale_result=whale_result,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
