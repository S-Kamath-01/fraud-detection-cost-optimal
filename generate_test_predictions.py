"""
generate_test_predictions.py

One-off script, NOT part of the permanent pipeline - generates a mix of
varied predictions against the running API to populate Postgres with
enough logged predictions (30+) to exercise the /drift-report endpoint
end-to-end. Run this manually when you want test data, then delete or
ignore it; it doesn't belong in src/.
"""

import requests
import random

BASE_URL = "http://localhost:8000"

random.seed(42)

transactions = []

# Low-value legitimate transactions
for _ in range(10):
    amount = random.uniform(50, 2000)
    old_balance = amount + random.uniform(1000, 50000)
    transactions.append({
        "amount": round(amount, 2),
        "oldbalanceOrg": round(old_balance, 2),
        "newbalanceOrig": round(old_balance - amount, 2),
        "type": random.choice(["TRANSFER", "CASH_OUT"]),
    })

# High-value legitimate transactions
for _ in range(10):
    amount = random.uniform(50000, 300000)
    old_balance = amount + random.uniform(100000, 500000)
    transactions.append({
        "amount": round(amount, 2),
        "oldbalanceOrg": round(old_balance, 2),
        "newbalanceOrig": round(old_balance - amount, 2),
        "type": random.choice(["TRANSFER", "CASH_OUT"]),
    })

# Fraud-like: account fully drained
for _ in range(10):
    amount = random.uniform(100, 2000000)
    transactions.append({
        "amount": round(amount, 2),
        "oldbalanceOrg": round(amount, 2),
        "newbalanceOrig": 0.0,
        "type": random.choice(["TRANSFER", "CASH_OUT"]),
    })

# Mixed / edge cases
for _ in range(10):
    amount = random.uniform(500, 100000)
    old_balance = random.uniform(0, 1000000)
    new_balance = max(0, old_balance - amount)
    transactions.append({
        "amount": round(amount, 2),
        "oldbalanceOrg": round(old_balance, 2),
        "newbalanceOrig": round(new_balance, 2),
        "type": random.choice(["TRANSFER", "CASH_OUT"]),
    })

random.shuffle(transactions)

print(f"Sending {len(transactions)} predictions to {BASE_URL}/predict...\n")

for i, txn in enumerate(transactions, 1):
    try:
        resp = requests.post(f"{BASE_URL}/predict", json=txn, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        print(f"{i:2}. amount={txn['amount']:>12,.2f}  type={txn['type']:<9}  "
              f"-> {result['decision']:<6} (p={result['probability']:.4f})")
    except requests.exceptions.RequestException as e:
        print(f"{i:2}. FAILED: {e}")

print(f"\nDone. Check /drift-report or the Streamlit dashboard now.")