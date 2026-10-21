# FIN451 Transaction Monitoring & Fraud Risk Scoring Prototype

A small version of the transaction-monitoring systems banks use for fraud and anti-money-laundering (AML) work. It:

1. **Generates** 5,000 realistic fake transactions for 50 customers, with 150 suspicious ones (3%) deliberately planted as an answer key.
2. **Scores** every transaction 0–100 using five transparent rules, and explains each flag in plain English.
3. **Displays** everything in an interactive dashboard where you can tune the rules and see how many planted anomalies were caught or missed.

## How to run

From this folder, in a terminal (PowerShell on Windows). These commands call the
virtual environment's Python directly, so you don't need to "activate" it. That
matters because Windows often blocks the activation script by default.

```bash
python -m venv .venv
```
```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
```
```bash
.venv\Scripts\python.exe generate_data.py
```
```bash
.venv\Scripts\python.exe -m streamlit run app.py
```

Stop the dashboard with **Ctrl+C** in the terminal.

The dashboard opens in your browser at http://localhost:8501. If `transactions.csv` doesn't exist yet, the app generates it automatically.

## Files

| File | What it does |
|---|---|
| `generate_data.py` | Creates `transactions.csv`. The fixed random seed (42) means everyone gets identical data. |
| `scoring.py` | Cleans the data, runs the five rules, computes risk scores, and grades results against the answer key. |
| `app.py` | The Streamlit dashboard. |
| `requirements.txt` | Python packages needed. |

## The five rules (default weights)

| Rule | Max points | Fires when |
|---|---|---|
| Unusual amount | 50 | The amount is far above this customer's typical amount for that transaction type (or, if they have fewer than 5 of that type, the typical amount across all customers): a robust z-score of 2+ on a log scale earns partial points, and 4.5+ earns full points |
| Velocity | 40 | The customer makes 4+ transactions within a 1-hour window |
| Late night | 25 | The transaction happens between 12:00 and 4:59 AM |
| New location / merchant | 40 | It's the customer's first time in that city **and** at that merchant (20 points for just one) |
| Structuring | 50 | A cash deposit is $9,000–$9,999.99, just under the $10,000 CTR threshold |

The default alert threshold is **40**. One strong signal (structuring, a burst, a very large amount, or a new city plus a new merchant) flags a transaction on its own. Weaker signals such as late-night timing need a second red flag.

## Default results on the synthetic data

| Planted type | Caught |
|---|---|
| Structuring | 30 / 30 |
| Rapid burst | 30 / 30 |
| New location | 28 / 30 |
| Large amount | 24 / 30 |
| Late night | 17 / 30 |
| **Total** | **129 / 150 (86% recall)**, with 30 false alarms (81% precision) |

Most misses are late-night card purchases only 1.5–3× the customer's normal amount. Each one looks almost normal, which is exactly the kind of case rule-based systems struggle with.

## Uploading your own data

The CSV needs at least `customer_id`, `timestamp`, and `amount`. `transaction_id`, `transaction_type`, `merchant`, and `location` are optional. Column names are matched loosely (for example, `Amt` → `amount` and `Date` → `timestamp`). Rows with bad dates, non-numeric amounts, or negative amounts are removed, and the app tells you how many were removed and why.

## Known simplifications

- Files with dates but no times of day switch off the late-night and velocity rules, since every transaction would otherwise look like it happened at midnight.
- A customer's "normal" is computed from all of their transactions in the file. A live system would only use past transactions.
- The velocity window looks 30 minutes before and after each transaction, which suits after-the-fact review rather than real-time blocking.
- The rule weights were tuned on this same synthetic data, so real-world performance would be lower.
