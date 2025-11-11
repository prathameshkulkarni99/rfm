DROP TABLE IF EXISTS rfm_table;

CREATE TABLE rfm_table AS
WITH customer_orders AS (
    SELECT
        customer_id,
        MAX(order_date) AS last_order_date,
        COUNT(order_id) AS frequency,
        SUM(amount) AS monetary
    FROM orders
    GROUP BY customer_id
)
SELECT
    customer_id,
    CAST(julianday('{REFERENCE_DATE}') - julianday(last_order_date) AS INTEGER) AS recency_days,
    frequency,
    monetary
FROM customer_orders;
