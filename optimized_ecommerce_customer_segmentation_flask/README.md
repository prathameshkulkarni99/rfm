# Optimized E-commerce Customer Segmentation (Flask Frontend)

Final-year project: **Optimized E-commerce Customer Segmentation using RFM Analysis with SQL Pushdown, K-Means vs Fuzzy C-Means Clustering and Blue Whale Optimization**, with a Flask web frontend.

## Features

1. Upload an e-commerce orders CSV (order_id, customer_id, order_date, amount).
2. Compute RFM features in SQLite using **SQL pushdown**.
3. Run **K-Means** clustering and pick the best k via silhouette score.
4. Run **Fuzzy C-Means (C-means)** clustering.
5. Run **Whale / Blue Whale Optimization** to optimize feature weights for RFM to maximize silhouette score.
6. View results (cluster summaries, sample labeled customers, fuzzy centers, optimization metrics) in a simple Flask UI.

## Project structure

```text
optimized_ecommerce_customer_segmentation_flask/
├── app.py                          # Flask app entry point
├── data/
│   └── raw_orders.csv              # Sample synthetic orders data
├── sql/
│   └── rfm_aggregation.sql         # SQL for RFM aggregation (SQL pushdown)
├── src/
│   └── main_pipeline.py            # Core pipeline functions
├── output/                         # Generated outputs (CSVs, plots, metrics)
├── templates/
│   ├── base.html
│   ├── index.html
│   └── results.html
├── static/
│   └── style.css
├── requirements.txt
└── README.md
```

## How to run (terminal)

```bash
cd optimized_ecommerce_customer_segmentation_flask
conda create -n ecommerce_rfm python=3.10 -y
conda activate ecommerce_rfm
pip install -r requirements.txt

# Option 1: run pipeline from terminal only
python -m src.main_pipeline

# Option 2: run Flask web app
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

From the web UI you can upload a CSV and see segmentation results.
