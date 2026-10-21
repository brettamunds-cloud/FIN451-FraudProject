"""
generate_data.py - Builds a synthetic (fake but realistic) bank transaction dataset.

WHY WE NEED THIS
Real bank data is confidential, so we simulate it. The big advantage of
simulated data is that WE decide which transactions are suspicious. That gives
us an "answer key" (the `injected_anomaly` column) to grade our fraud detector.

HOW IT WORKS
1. Create 50 customers, each with their own habits: home city, favorite
   merchants, typical spending level, the hours they're usually awake, and how
   often they use cards vs. ATMs vs. deposits vs. transfers.
2. Generate ~4,850 "normal" transactions that follow each customer's habits.
3. Plant ~150 suspicious transactions (3%) of five kinds:
   large amounts, rapid bursts, late-night activity, new locations, and
   structuring (cash deposits just under the $10,000 reporting threshold).
4. Save everything to transactions.csv.

Run it with:   python generate_data.py
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Settings - change these to experiment
# ---------------------------------------------------------------------------
SEED = 42                      # Fixed seed = same "random" data every run
N_CUSTOMERS = 50
TOTAL_ROWS = 5000
ANOMALIES_PER_TYPE = 30        # 5 types x 30 = 150 rows = 3% of 5,000
START_DATE = pd.Timestamp("2026-07-01")
N_DAYS = 90
OUTPUT_FILE = "transactions.csv"

CITIES = [
    "Chicago, IL", "Milwaukee, WI", "Indianapolis, IN", "Detroit, MI",
    "Columbus, OH", "St. Louis, MO", "Minneapolis, MN", "Des Moines, IA",
    "Kansas City, MO", "Louisville, KY", "Cincinnati, OH", "Madison, WI",
]
# Far-away places our customers never normally visit (used for "new location" fraud)
FAR_CITIES = ["Miami, FL", "Las Vegas, NV", "Los Angeles, CA", "Lagos, NG",
              "London, UK", "Houston, TX", "Phoenix, AZ", "New York, NY"]

EVERYDAY_MERCHANTS = [
    "Jewel-Osco", "Kroger", "Target", "Walmart", "Costco", "Aldi", "Trader Joe's",
    "Walgreens", "CVS Pharmacy", "Shell", "BP", "Speedway", "Starbucks",
    "Dunkin'", "Chipotle", "McDonald's", "Panera Bread", "Subway", "Home Depot",
    "Best Buy", "Amazon", "Netflix", "Spotify", "Uber", "Lyft", "DoorDash",
    "Old Navy", "Kohl's", "PetSmart", "Planet Fitness", "AMC Theatres",
    "Barnes & Noble", "Comcast Xfinity", "ComEd Electric", "Verizon Wireless",
]
# Merchants associated with fraud (gift cards, resellable electronics, etc.)
RISKY_MERCHANTS = ["Gift Card Kiosk", "Electronics Depot Online", "Luxury Watch Boutique",
                   "Crypto ATM", "Prepaid Card Center", "Overseas Online Retailer"]
BANKS = ["Chase", "Bank of America", "PNC", "Fifth Third"]
TRANSACTION_TYPES = ["card purchase", "ATM withdrawal", "cash deposit", "transfer"]


# ---------------------------------------------------------------------------
# Step 1: customer profiles ("what is normal for this person?")
# ---------------------------------------------------------------------------
def build_customer_profiles(rng, n_customers=N_CUSTOMERS):
    """Create one dictionary of habits per customer."""
    profiles = []
    for i in range(1, n_customers + 1):
        home, *others = rng.choice(CITIES, size=3, replace=False)
        n_secondary = rng.integers(1, 3)              # 1 or 2 other cities they visit
        n_regulars = rng.integers(5, 9)               # 5-8 favorite merchants
        wake_hour = int(rng.integers(6, 10))          # first activity 6-9 AM
        sleep_hour = int(rng.integers(21, 24))        # last activity 9-11 PM

        # Mix of transaction types: mostly card purchases, like a real person
        card_share = rng.uniform(0.70, 0.85)
        rest = rng.dirichlet([3, 1.5, 2]) * (1 - card_share)   # ATM, deposit, transfer
        type_mix = [card_share, *rest]

        profiles.append({
            "customer_id": f"C{i:03d}",
            "home_city": home,
            "secondary_cities": list(others[:n_secondary]),
            "regular_merchants": list(rng.choice(EVERYDAY_MERCHANTS, size=n_regulars, replace=False)),
            # Lognormal = most people spend modestly, a few spend a lot (like real incomes)
            "typical_card_spend": float(np.clip(rng.lognormal(np.log(55), 0.5), 15, 250)),
            "typical_transfer": float(np.clip(rng.lognormal(np.log(350), 0.6), 50, 2500)),
            "typical_deposit": float(np.clip(rng.lognormal(np.log(700), 0.5), 100, 3000)),
            "bank": str(rng.choice(BANKS)),
            "transfer_apps": list(rng.choice(["Zelle Transfer", "ACH Transfer", "Venmo Transfer"],
                                             size=rng.integers(1, 3), replace=False)),
            "active_hours": (wake_hour, sleep_hour),
            "type_mix": type_mix,
        })
    return profiles


# ---------------------------------------------------------------------------
# Step 2: normal transactions
# ---------------------------------------------------------------------------
def random_normal_timestamp(rng, profile, day_window=(0, N_DAYS)):
    """A timestamp during the customer's usual waking hours."""
    day = int(rng.integers(*day_window))
    wake, sleep = profile["active_hours"]
    hour = int(rng.integers(wake, sleep))
    # About 1% of honest transactions happen late at night anyway (insomnia, night shifts),
    # so the late-night rule will produce some false alarms - just like in real life.
    if rng.random() < 0.01:
        hour = int(rng.integers(0, 5))
    return START_DATE + pd.Timedelta(days=day, hours=hour, minutes=int(rng.integers(0, 60)),
                                     seconds=int(rng.integers(0, 60)))


def make_normal_transaction(rng, profile, timestamp):
    """One ordinary transaction that fits the customer's habits."""
    tx_type = str(rng.choice(TRANSACTION_TYPES, p=profile["type_mix"]))
    home = profile["home_city"]
    # 85% of activity happens in the home city, the rest in their other usual cities
    location = home if rng.random() < 0.85 else str(rng.choice(profile["secondary_cities"]))

    if tx_type == "card purchase":
        # 95% of the time at a favorite merchant; 5% they try somewhere new
        if rng.random() < 0.95:
            merchant = str(rng.choice(profile["regular_merchants"]))
        else:
            merchant = str(rng.choice(EVERYDAY_MERCHANTS))
        amount = rng.lognormal(np.log(profile["typical_card_spend"]), 0.5)
    elif tx_type == "ATM withdrawal":
        merchant = f"{profile['bank']} ATM"
        amount = rng.choice([20, 40, 60, 80, 100, 120, 200], p=[.10, .20, .20, .15, .20, .05, .10])
    elif tx_type == "cash deposit":
        merchant = f"{profile['bank']} Branch Deposit"
        location = home
        amount = rng.lognormal(np.log(profile["typical_deposit"]), 0.4)
    else:  # transfer
        merchant = str(rng.choice(profile["transfer_apps"]))
        amount = rng.lognormal(np.log(profile["typical_transfer"]), 0.5)

    return {
        "customer_id": profile["customer_id"], "timestamp": timestamp,
        "amount": round(float(amount), 2), "transaction_type": tx_type,
        "merchant": merchant, "location": location,
        "injected_anomaly": False, "anomaly_type": "none",
    }


def allocate_rows(rng, n_rows, n_customers):
    """Split the normal rows across customers - some customers are busier than others."""
    activity = rng.uniform(0.5, 1.5, size=n_customers)
    counts = np.floor(activity / activity.sum() * n_rows).astype(int)
    counts[: n_rows - counts.sum()] += 1          # hand out leftover rows from rounding
    return counts


def generate_normal_transactions(rng, profiles, n_rows):
    """Generate ordinary activity for every customer.

    The LAST customer is a brand-new account with only 3 transactions. This
    deliberately tests how the detector handles customers with too little history.
    """
    rows = []
    newcomer = profiles[-1]
    for _ in range(3):
        rows.append(make_normal_transaction(
            rng, newcomer, random_normal_timestamp(rng, newcomer, (N_DAYS - 7, N_DAYS))))

    counts = allocate_rows(rng, n_rows - 3, len(profiles) - 1)
    for profile, count in zip(profiles[:-1], counts):
        for _ in range(count):
            rows.append(make_normal_transaction(rng, profile, random_normal_timestamp(rng, profile)))
    return rows


# ---------------------------------------------------------------------------
# Step 3: plant suspicious transactions (our "answer key")
# ---------------------------------------------------------------------------
def _anomaly(profile, timestamp, amount, tx_type, merchant, location, kind):
    return {
        "customer_id": profile["customer_id"], "timestamp": timestamp,
        "amount": round(float(amount), 2), "transaction_type": tx_type,
        "merchant": merchant, "location": location,
        "injected_anomaly": True, "anomaly_type": kind,
    }


def inject_large_amounts(rng, profiles, n):
    """Purchases or transfers 6-15x bigger than the customer normally spends."""
    rows = []
    for profile in rng.choice(profiles, size=n):
        ts = random_normal_timestamp(rng, profile, (14, N_DAYS))
        multiple = rng.uniform(6, 15)
        if rng.random() < 0.6:
            rows.append(_anomaly(profile, ts, profile["typical_card_spend"] * multiple, "card purchase",
                                 str(rng.choice(profile["regular_merchants"])), profile["home_city"],
                                 "large_amount"))
        else:
            rows.append(_anomaly(profile, ts, profile["typical_transfer"] * multiple, "transfer",
                                 profile["transfer_apps"][0], profile["home_city"], "large_amount"))
    return rows


def inject_rapid_bursts(rng, profiles, n, burst_size=5):
    """Several card swipes within ~30 minutes - like a thief testing a stolen card."""
    rows = []
    for profile in rng.choice(profiles, size=n // burst_size, replace=False):
        start = random_normal_timestamp(rng, profile, (14, N_DAYS))
        offsets = np.sort(rng.integers(0, 30 * 60, size=burst_size))   # seconds within 30 min
        for offset in offsets:
            rows.append(_anomaly(profile, start + pd.Timedelta(seconds=int(offset)),
                                 rng.lognormal(np.log(profile["typical_card_spend"]), 0.4),
                                 "card purchase", str(rng.choice(EVERYDAY_MERCHANTS)),
                                 profile["home_city"], "rapid_burst"))
    return rows


def inject_late_night(rng, profiles, n):
    """Activity between midnight and 5 AM.

    Half are ATM withdrawals well above the customer's usual cash amounts (a stolen
    card being drained); half are subtler card purchases only 1.5-3x normal. The
    subtle ones are realistic "hard cases" that a rule-based system may miss.
    """
    rows = []
    for i, profile in enumerate(rng.choice(profiles, size=n)):
        ts = START_DATE + pd.Timedelta(days=int(rng.integers(14, N_DAYS)), hours=int(rng.integers(0, 5)),
                                       minutes=int(rng.integers(0, 60)))
        if i % 2 == 0:
            rows.append(_anomaly(profile, ts, rng.choice([400, 500, 600, 800]), "ATM withdrawal",
                                 f"{profile['bank']} ATM", profile["home_city"], "late_night"))
        else:
            rows.append(_anomaly(profile, ts, profile["typical_card_spend"] * rng.uniform(1.5, 3),
                                 "card purchase", str(rng.choice(profile["regular_merchants"])),
                                 profile["home_city"], "late_night"))
    return rows


def inject_new_locations(rng, profiles, n):
    """A purchase in a city the customer has never visited, at a merchant they've never used."""
    rows = []
    for profile in rng.choice(profiles, size=n):
        ts = random_normal_timestamp(rng, profile, (14, N_DAYS))
        rows.append(_anomaly(profile, ts, profile["typical_card_spend"] * rng.uniform(1, 4),
                             "card purchase", str(rng.choice(RISKY_MERCHANTS)),
                             str(rng.choice(FAR_CITIES)), "new_location"))
    return rows


def inject_structuring(rng, profiles, n, deposits_per_customer=5):
    """Cash deposits of $9,000-$9,999 spread over a couple of weeks.

    US banks must file a Currency Transaction Report (CTR) for cash over $10,000.
    Deliberately keeping deposits just below that line to dodge the report is
    called "structuring" and is itself a federal crime (31 U.S.C. 5324).
    """
    rows = []
    for profile in rng.choice(profiles, size=n // deposits_per_customer, replace=False):
        first_day = int(rng.integers(14, N_DAYS - 15))
        for _ in range(deposits_per_customer):
            ts = random_normal_timestamp(rng, profile, (first_day, first_day + 14))
            rows.append(_anomaly(profile, ts, rng.uniform(9000, 9999.99), "cash deposit",
                                 f"{profile['bank']} Branch Deposit", profile["home_city"],
                                 "structuring"))
    return rows


def inject_anomalies(rng, profiles, n_per_type=ANOMALIES_PER_TYPE):
    """Plant all five kinds of suspicious activity. The brand-new account is left alone."""
    eligible = profiles[:-1]
    return (inject_large_amounts(rng, eligible, n_per_type)
            + inject_rapid_bursts(rng, eligible, n_per_type)
            + inject_late_night(rng, eligible, n_per_type)
            + inject_new_locations(rng, eligible, n_per_type)
            + inject_structuring(rng, eligible, n_per_type))


# ---------------------------------------------------------------------------
# Step 4: put it all together
# ---------------------------------------------------------------------------
def main(output_path=OUTPUT_FILE, seed=SEED):
    """Generate the dataset, save it to CSV, and return it as a DataFrame."""
    rng = np.random.default_rng(seed)
    profiles = build_customer_profiles(rng)

    anomalies = inject_anomalies(rng, profiles)
    normal = generate_normal_transactions(rng, profiles, TOTAL_ROWS - len(anomalies))

    df = pd.DataFrame(normal + anomalies).sort_values("timestamp").reset_index(drop=True)
    df.insert(0, "transaction_id", [f"TXN{i:05d}" for i in range(1, len(df) + 1)])
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df):,} transactions for {df['customer_id'].nunique()} customers to {output_path}")
    print(f"Injected anomalies: {df['injected_anomaly'].sum()} ({df['injected_anomaly'].mean():.1%})")
    print(df.loc[df["injected_anomaly"], "anomaly_type"].value_counts().to_string())
    return df


if __name__ == "__main__":
    main()
