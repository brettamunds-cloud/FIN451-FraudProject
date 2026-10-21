"""
scoring.py - The risk-scoring "brain" of the fraud monitor.

This file does three jobs:

1. CLEAN THE DATA  (load_and_validate)
   Real-world files are messy: columns named "Amt" instead of "amount", dates
   that aren't dates, negative amounts, blank cells. We fix what we safely can,
   drop what we can't, and report every change as a plain-English warning.

2. RUN FIVE RULES  (compute_rule_signals)
   Each rule looks at one red flag and gives every transaction a "signal
   strength" from 0 (no concern) to 1 (strong concern) plus a reason:
     - Unusual amount   : much bigger than this customer usually spends
     - Velocity         : many transactions packed into one hour
     - Late night       : between 12:00 and 4:59 AM
     - New place        : first time at this city and/or merchant
     - Structuring      : cash deposit of $9,000-$9,999.99

3. TURN SIGNALS INTO A SCORE  (score_transactions)
   Each rule has a weight = the maximum points it can add. The risk score is
   the sum of (weight x strength), capped at 100. Anything at or above the
   threshold is flagged for an analyst to review.

Rule-based scoring is how many banks' transaction-monitoring systems work.
It's transparent: every flag comes with a reason a compliance officer can read
and defend to a regulator. A machine-learning model might catch more, but it
is much harder to explain.
"""

import io

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Columns we cannot work without vs. columns we can fill in if missing
REQUIRED_COLUMNS = ["customer_id", "timestamp", "amount"]
OPTIONAL_COLUMNS = ["transaction_id", "transaction_type", "merchant", "location"]

# Common alternative names people use for each column (all lowercase, underscores)
COLUMN_ALIASES = {
    "transaction_id": ["txn_id", "trans_id", "id", "transaction", "transaction_number", "txn"],
    "customer_id": ["customer", "cust_id", "customer_number", "client_id", "account_id", "account"],
    "timestamp": ["date", "datetime", "time", "date_time", "transaction_date", "trans_date", "posted_date"],
    "amount": ["amt", "transaction_amount", "value", "amount_usd", "dollar_amount"],
    "transaction_type": ["type", "txn_type", "trans_type", "category", "channel"],
    "merchant": ["merchant_name", "payee", "vendor", "description", "counterparty"],
    "location": ["city", "place", "merchant_city", "transaction_location"],
    "injected_anomaly": ["is_fraud", "fraud", "label", "is_anomaly", "anomaly"],
}

# Map messy transaction-type text onto our four standard types
TYPE_KEYWORDS = [
    ("atm", "ATM withdrawal"),
    ("withdraw", "ATM withdrawal"),
    ("deposit", "cash deposit"),
    ("transfer", "transfer"),
    ("wire", "transfer"),
    ("zelle", "transfer"),
    ("card", "card purchase"),
    ("purchase", "card purchase"),
    ("pos", "card purchase"),
]

# Rule names, display labels, and default weights (max points each rule can add)
RULES = ["amount", "velocity", "late_night", "new_place", "structuring"]
RULE_LABELS = {
    "amount": "Unusual amount",
    "velocity": "High velocity (1-hour window)",
    "late_night": "Late night (12-5 AM)",
    "new_place": "New location / merchant",
    "structuring": "Possible structuring",
}
DEFAULT_WEIGHTS = {"amount": 50, "velocity": 40, "late_night": 25, "new_place": 40, "structuring": 50}
DEFAULT_THRESHOLD = 40

# Rule parameters
MIN_HISTORY = 5                 # transactions needed before we trust a customer's "normal"
Z_FIRE = 2.0                    # amount rule gives partial points from this z-score...
Z_FULL = 4.5                    # ...and reaches full strength here
VELOCITY_WINDOW_MIN = 60        # length of the velocity window, in minutes
VELOCITY_MIN_COUNT = 4          # this many transactions in the window = suspicious
LATE_NIGHT_END_HOUR = 5         # 00:00-04:59 counts as late night
NEW_PLACE_WARMUP = 5            # ignore a customer's first N transactions (everything is "new")
STRUCTURING_LOW, STRUCTURING_HIGH = 9_000, 10_000
CTR_THRESHOLD = 10_000          # Currency Transaction Report threshold


class DataValidationError(Exception):
    """Raised when a file can't be used at all. The message is shown to the user."""


# ---------------------------------------------------------------------------
# 1. Loading and cleaning
# ---------------------------------------------------------------------------
def read_transactions_csv(source):
    """Read a CSV from a file path, file-like object, or raw bytes, with friendly errors."""
    if isinstance(source, (bytes, bytearray)):
        if not source.strip():
            raise DataValidationError("The uploaded file is empty.")
        source = io.BytesIO(source)
    try:
        df = pd.read_csv(source)
        # Excel in some regions saves "CSV" files with semicolons or tabs between columns.
        # If everything landed in one column, let pandas detect the separator instead.
        if df.shape[1] == 1 and any(sep in str(df.columns[0]) for sep in [";", "\t", "|"]):
            if hasattr(source, "seek"):
                source.seek(0)
            df = pd.read_csv(source, sep=None, engine="python")
        return df
    except pd.errors.EmptyDataError:
        raise DataValidationError("The file is empty - it has no column headers or rows.")
    except UnicodeDecodeError:
        raise DataValidationError("The file doesn't look like a text CSV. If it's an Excel "
                                  "file, use File > Save As > CSV first.")
    except pd.errors.ParserError as e:
        raise DataValidationError(f"The file couldn't be read as a CSV ({e}). Check that every "
                                  "row has the same number of commas.")


def standardize_column_names(df):
    """Rename columns to our standard names, e.g. 'Amt ' -> 'amount', 'Date' -> 'timestamp'.

    Returns the renamed DataFrame and a list of the renames we made.
    """
    cleaned = {col: str(col).strip().lower().replace(" ", "_").replace("-", "_") for col in df.columns}
    df = df.rename(columns=cleaned)

    renames = {}
    for standard, aliases in COLUMN_ALIASES.items():
        if standard in df.columns:
            continue
        for alias in aliases:
            if alias in df.columns and alias not in renames:
                renames[alias] = standard
                break
    df = df.rename(columns=renames)
    notes = [f"Treated column '{old}' as '{new}'." for old, new in renames.items()]

    # Two columns can end up with the same name (e.g. "Customer ID" and "customer_id").
    # Keep the first one so later steps don't get confused.
    duplicated = df.columns.duplicated()
    if duplicated.any():
        dupes = sorted(set(df.columns[duplicated]))
        notes.append(f"Found more than one column named {', '.join(dupes)} - used the first one.")
        df = df.loc[:, ~duplicated]
    return df, notes


def parse_timestamps(values):
    """Convert text to dates/times. Returns (timestamps, note_or_None).

    Unreadable values become NaT ("not a time") so they can be dropped. Time-zone
    labels like "Z" or "-05:00" are ignored and the local clock time is kept,
    because "late night" means late night where the customer was.
    """
    try:
        parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        # Mixed time zones (or tz-aware mixed with plain times): parse one by one
        parsed = pd.Series([pd.to_datetime(v, errors="coerce") for v in values], index=values.index)
        parsed = parsed.map(lambda t: t.tz_localize(None) if pd.notna(t) and t.tzinfo else t)
        return pd.to_datetime(parsed, errors="coerce"), "Time-zone labels were ignored; local clock times were used."
    if getattr(parsed.dt, "tz", None) is not None:
        return parsed.dt.tz_localize(None), "Time-zone labels were ignored; local clock times were used."
    return parsed, None


def normalize_transaction_type(value):
    """Map free-text types like 'ATM', 'Cash Deposit', 'POS purchase' to our four standard types."""
    text = str(value).strip().lower()
    for keyword, standard in TYPE_KEYWORDS:
        if keyword in text:
            return standard
    return text if text and text != "nan" else "unknown"


def parse_bool(series):
    """Turn True/False, 1/0, yes/no text into real booleans."""
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "t"])


def load_and_validate(raw):
    """Check and clean a raw transactions table.

    Returns (clean_df, warnings). Raises DataValidationError if the data is unusable.
    """
    warnings = []
    if raw is None or raw.shape[1] == 0 or raw.empty:
        raise DataValidationError("The file has no transaction rows.")

    df, rename_notes = standardize_column_names(raw.copy())
    warnings += rename_notes

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Columns found in your file: {', '.join(map(str, raw.columns))}. "
            "Rename your columns to match (customer_id, timestamp, amount are required)."
        )

    # Optional columns: fill in so the app keeps working, but say which rules are weakened
    if "transaction_id" not in df.columns:
        df["transaction_id"] = [f"ROW{i:05d}" for i in range(1, len(df) + 1)]
        warnings.append("No transaction_id column - generated IDs automatically.")
    for col, consequence in [
        ("transaction_type", "the structuring rule can't run and amounts are compared across all types"),
        ("merchant", "the new-merchant check can't run"),
        ("location", "the new-location check can't run"),
    ]:
        if col not in df.columns:
            df[col] = "Unknown"
            warnings.append(f"No '{col}' column - filled with 'Unknown', so {consequence}.")

    start_rows = len(df)

    # Customer IDs: drop blanks
    # (A numeric ID column with blanks gets read as decimals, e.g. 1001 -> "1001.0"; undo that.)
    df["customer_id"] = (df["customer_id"].astype("string").str.strip()
                         .str.replace(r"^(\d+)\.0$", r"\1", regex=True))
    bad_customer = df["customer_id"].isna() | (df["customer_id"] == "")
    if bad_customer.any():
        warnings.append(f"Removed {bad_customer.sum()} row(s) with a blank customer_id.")
        df = df[~bad_customer]

    # Timestamps: anything that can't be read as a date becomes NaT and is dropped
    df["timestamp"], tz_note = parse_timestamps(df["timestamp"])
    if tz_note:
        warnings.append(tz_note)
    bad_dates = df["timestamp"].isna()
    if bad_dates.any():
        warnings.append(f"Removed {bad_dates.sum()} row(s) with a missing or unreadable date.")
        df = df[~bad_dates]
    odd_years = ~df["timestamp"].dt.year.between(1990, 2100)
    if odd_years.any():
        warnings.append(f"{odd_years.sum()} row(s) have dates before 1990 or after 2100 "
                        f"(e.g. {df.loc[odd_years, 'timestamp'].iloc[0]:%Y-%m-%d}). "
                        "Check the timestamp column; they were kept but may be wrong.")

    # Amounts: strip "$" and "," then convert; drop anything that still isn't a real number
    df["amount"] = pd.to_numeric(
        df["amount"].astype(str).str.replace(r"[$,\s]", "", regex=True), errors="coerce")
    bad_amounts = df["amount"].isna() | np.isinf(df["amount"])
    if bad_amounts.any():
        warnings.append(f"Removed {bad_amounts.sum()} row(s) with a missing or non-numeric amount.")
        df = df[~bad_amounts]
    negative = df["amount"] < 0
    if negative.any():
        warnings.append(f"Removed {negative.sum()} row(s) with negative amounts (likely refunds or "
                        "reversals, which this monitor doesn't score).")
        df = df[~negative]

    # Text columns: fill blanks, standardize transaction types
    for col in ["merchant", "location"]:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    df["transaction_type"] = df["transaction_type"].map(normalize_transaction_type)
    df["transaction_id"] = df["transaction_id"].astype(str)

    # Answer-key column (only exists for test data)
    if "injected_anomaly" in df.columns:
        df["injected_anomaly"] = parse_bool(df["injected_anomaly"])
    if "anomaly_type" in df.columns:
        df["anomaly_type"] = df["anomaly_type"].fillna("unknown").astype(str)

    if df.empty:
        raise DataValidationError(f"All {start_rows} row(s) were invalid after cleaning, so there's "
                                  "nothing left to score. Problems found: "
                                  + " ".join(w for w in warnings if w.startswith("Removed")))

    # Customers with too little history: still scored, but some rules skip them
    counts = df["customer_id"].value_counts()
    thin = counts[counts < MIN_HISTORY]
    if len(thin):
        examples = ", ".join(thin.index[:5])
        warnings.append(f"{len(thin)} customer(s) have fewer than {MIN_HISTORY} transactions "
                        f"(e.g. {examples}). The unusual-amount and new-place rules are skipped "
                        "for them because we don't yet know what's normal.")

    if not has_time_of_day(df):
        warnings.append("Timestamps contain dates only (no times of day), so the late-night "
                        "and velocity rules were switched off.")

    df = df.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)
    return df, warnings


# ---------------------------------------------------------------------------
# 2. The five rules
#    Each returns (strength, reason): two Series aligned with df.
#    strength is 0-1; reason is a plain-English sentence ("" if the rule didn't fire).
# ---------------------------------------------------------------------------
def _reason_series(df, fired, texts):
    """Build the reason column: the given texts where the rule fired, "" elsewhere."""
    reason = pd.Series("", index=df.index, dtype=object)
    reason.loc[fired] = pd.Series(list(texts), index=df.index[fired], dtype=object)
    return reason


def has_time_of_day(df):
    """False when every timestamp is exactly midnight, i.e. the file only has dates.

    Many bank exports only include the posting date. Without real times, every
    transaction would look like it happened at 12:00 AM (late night!) and four
    purchases on the same day would look like a one-hour burst.
    """
    return bool((df["timestamp"] != df["timestamp"].dt.normalize()).any())


def rule_unusual_amount(df):
    """Rule 1: How unusual is this amount *for this customer*?

    We measure "unusual" with a robust z-score:
        z = (amount - customer's median) / (customer's typical spread)

    Points start at z = Z_FIRE (somewhat unusual - partial points, needs another
    red flag to trigger a review) and max out at z = Z_FULL (flags on its own).

    Two finance-friendly tweaks:
    * We compare on a LOG scale, i.e. in percentage terms. Going from $50 to $500
      is as unusual as going from $500 to $5,000 - both are 10x.
    * We use the MEDIAN and MAD (median absolute deviation) instead of the mean and
      standard deviation. One giant fraudulent charge would inflate a customer's
      mean and std and hide itself; the median barely moves.

    Each transaction is compared with the same customer's transactions of the same
    type (card purchases vs. card purchases, etc.), because a $500 transfer is
    normal even for someone whose coffee is $5. If a customer has fewer than
    MIN_HISTORY transactions of that type (say, only two transfers ever), we
    compare against that type across ALL customers instead (a "peer group").
    Comparing their transfer with their own coffee purchases would wrongly
    make every ordinary transfer look huge.

    Simplification: we use the customer's full history in the file. A live system
    would only use transactions from *before* the one being scored.
    """
    log_amt = np.log1p(df["amount"])

    def robust_stats(keys):
        grouped = log_amt.groupby([df[k] for k in keys])
        median = grouped.transform("median")
        mad = (log_amt - median).abs().groupby([df[k] for k in keys]).transform("median")
        count = grouped.transform("size")
        return median, mad, count

    med_own, mad_own, n_own = robust_stats(["customer_id", "transaction_type"])
    med_peer, mad_peer, n_peer = robust_stats(["transaction_type"])
    med_all, mad_all, n_all = robust_stats(["customer_id"])

    # Pick the best available baseline for each transaction:
    # 1) this customer's same type  2) all customers' same type  3) this customer's everything
    use_own = n_own >= MIN_HISTORY
    use_peer = ~use_own & (n_peer >= MIN_HISTORY)
    median = med_own.where(use_own, med_peer.where(use_peer, med_all))
    mad = mad_own.where(use_own, mad_peer.where(use_peer, mad_all))
    enough_history = n_all >= MIN_HISTORY

    # 1.4826 x MAD approximates a standard deviation. The floor (0.15 = ~15%) stops
    # customers with near-identical amounts (always $100 at the ATM) from dividing by ~0.
    spread = np.maximum(1.4826 * mad, 0.15)
    z = ((log_amt - median) / spread).where(enough_history, 0.0)

    fired = z >= Z_FIRE
    strength = (z / Z_FULL).clip(0, 1).where(fired, 0.0)

    typical = np.expm1(median)
    baseline = np.where(use_own, "this customer's typical " + df["transaction_type"],
                        np.where(use_peer, "the typical " + df["transaction_type"] + " across all customers",
                                 "this customer's typical transaction"))
    texts = [f"Amount ${df.at[i, 'amount']:,.2f} is unusually large - {baseline[k]} is about "
             f"${typical[i]:,.2f} (robust z-score {z[i]:.1f}; full points at {Z_FULL:g})."
             for k, i in zip(np.flatnonzero(fired), df.index[fired])]
    return strength, _reason_series(df, fired, texts)


def rule_velocity(df):
    """Rule 2: Are many transactions packed into a short time?

    For each transaction we count how many of the customer's transactions fall in
    the 1-hour window centred on it (30 minutes either side). Four or more in an
    hour is a classic sign of a stolen card being used quickly before it's blocked.

    (A real-time system could only look backwards; looking both ways lets the first
    swipe of a burst get flagged too, which suits after-the-fact review.)
    """
    if not has_time_of_day(df):
        return pd.Series(0.0, index=df.index), pd.Series("", index=df.index, dtype=object)

    half_window = VELOCITY_WINDOW_MIN * 60 // 2                         # in seconds
    counts = pd.Series(0, index=df.index)
    for _, idx in df.groupby("customer_id").groups.items():
        # Times as whole seconds since 1970 (df is already sorted by time within customer)
        times = df.loc[idx, "timestamp"].to_numpy().astype("datetime64[s]").astype("int64")
        lo = np.searchsorted(times, times - half_window, side="left")
        hi = np.searchsorted(times, times + half_window, side="right")
        counts[idx] = hi - lo

    fired = counts >= VELOCITY_MIN_COUNT
    strength = fired.astype(float)
    texts = [f"{n} transactions by this customer within one hour "
             f"(rule fires at {VELOCITY_MIN_COUNT})." for n in counts[fired]]
    return strength, _reason_series(df, fired, texts)


def rule_late_night(df):
    """Rule 3: Did it happen between midnight and 5 AM, when most people are asleep?"""
    if not has_time_of_day(df):
        return pd.Series(0.0, index=df.index), pd.Series("", index=df.index, dtype=object)

    fired = df["timestamp"].dt.hour < LATE_NIGHT_END_HOUR
    strength = fired.astype(float)
    texts = [f"Made at {t:%I:%M %p}, during late-night hours (12-5 AM)."
             for t in df.loc[fired, "timestamp"]]
    return strength, _reason_series(df, fired, texts)


def rule_new_place(df):
    """Rule 4: Is this the customer's first time in this city or at this merchant?

    A new merchant OR a new city alone is half-strength (people do try new
    restaurants). Both at once - e.g. a gift-card store in another state - is
    full strength. A customer's first few transactions are skipped because
    everything looks "new" for a new account.
    """
    nth_txn = df.groupby("customer_id").cumcount()                 # 0 = customer's first
    past_warmup = nth_txn >= NEW_PLACE_WARMUP
    new_location = (df.groupby(["customer_id", "location"]).cumcount() == 0) \
        & (df["location"] != "Unknown") & past_warmup
    new_merchant = (df.groupby(["customer_id", "merchant"]).cumcount() == 0) \
        & (df["merchant"] != "Unknown") & past_warmup

    strength = 0.5 * new_location.astype(float) + 0.5 * new_merchant.astype(float)
    fired = new_location | new_merchant
    texts = []
    for i in df.index[fired]:
        parts = []
        if new_location[i]:
            parts.append(f"in {df.at[i, 'location']}")
        if new_merchant[i]:
            parts.append(f"at {df.at[i, 'merchant']}")
        texts.append(f"First time this customer has transacted {' and '.join(parts)}.")
    return strength, _reason_series(df, fired, texts)


def rule_structuring(df):
    """Rule 5: A cash deposit just under the $10,000 reporting threshold.

    Banks must file a Currency Transaction Report (CTR) for cash over $10,000.
    Splitting cash into deposits just below that line to avoid the report is
    called structuring, and it's illegal even if the money itself is clean.
    """
    fired = (df["transaction_type"] == "cash deposit") \
        & df["amount"].between(STRUCTURING_LOW, STRUCTURING_HIGH, inclusive="left")
    strength = fired.astype(float)

    # Extra context: how many such deposits has this customer made in total?
    per_customer = fired.groupby(df["customer_id"]).transform("sum")
    texts = [
        f"Cash deposit of ${amt:,.2f} is just under the ${CTR_THRESHOLD:,} CTR reporting threshold"
        + (f" (one of {n} such deposits by this customer)." if n > 1 else ".")
        for amt, n in zip(df.loc[fired, "amount"], per_customer[fired])
    ]
    return strength, _reason_series(df, fired, texts)


RULE_FUNCTIONS = {
    "amount": rule_unusual_amount,
    "velocity": rule_velocity,
    "late_night": rule_late_night,
    "new_place": rule_new_place,
    "structuring": rule_structuring,
}


def compute_rule_signals(df):
    """Run every rule once and attach its strength (sig_*) and reason (why_*) columns.

    This is the slow part, so the app runs it once and re-uses the result
    whenever the user moves a weight or threshold slider.
    """
    out = df.copy()
    for name, rule in RULE_FUNCTIONS.items():
        out[f"sig_{name}"], out[f"why_{name}"] = rule(out)

    # A note for customers we can't fully judge
    history = out.groupby("customer_id")["customer_id"].transform("size")
    out["note"] = np.where(history < MIN_HISTORY,
                           "Note: only " + history.astype(str) + " transaction(s) on file for this "
                           "customer, so amount and new-place checks were skipped.", "")
    return out


# ---------------------------------------------------------------------------
# 3. Scoring
# ---------------------------------------------------------------------------
def score_transactions(signals, weights=None, threshold=DEFAULT_THRESHOLD):
    """Combine rule signals into a 0-100 risk score, a flag, and an explanation.

    signals   : output of compute_rule_signals
    weights   : dict of rule -> max points (defaults to DEFAULT_WEIGHTS)
    threshold : scores at or above this are flagged
    """
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    out = signals.copy()

    for name in RULES:
        out[f"pts_{name}"] = (out[f"sig_{name}"] * weights[name]).round(1)
    out["risk_score"] = out[[f"pts_{n}" for n in RULES]].sum(axis=1).clip(upper=100).round(1)
    out["flagged"] = out["risk_score"] >= threshold
    out["rules_fired"] = out.apply(
        lambda r: ", ".join(RULE_LABELS[n] for n in RULES if r[f"pts_{n}"] > 0), axis=1)
    out["explanation"] = out.apply(lambda r: build_explanation(r, threshold), axis=1)
    return out


def build_explanation(row, threshold):
    """Write a readable explanation, e.g. '[+45 pts] Amount $4,820 is unusually large...'"""
    parts = [f"[+{row[f'pts_{n}']:g} pts] {row[f'why_{n}']}" for n in RULES if row[f"pts_{n}"] > 0]
    if not parts:
        text = "No rules fired."
    else:
        verdict = "FLAGGED" if row["risk_score"] >= threshold else "Below threshold"
        text = f"{verdict} (score {row['risk_score']:g} vs. threshold {threshold:g}): " + " ".join(parts)
    if row.get("note"):
        text += " " + row["note"]
    return text


# ---------------------------------------------------------------------------
# 4. Grading the detector against the answer key
# ---------------------------------------------------------------------------
def evaluate_detection(scored):
    """Compare flags with the injected_anomaly answer key.

    Returns None if the data has no answer key (e.g. a user's own upload).
    """
    if "injected_anomaly" not in scored.columns:
        return None

    truth, flagged = scored["injected_anomaly"], scored["flagged"]
    caught = int((truth & flagged).sum())          # true positives
    missed = int((truth & ~flagged).sum())         # false negatives
    false_alarms = int((~truth & flagged).sum())   # false positives
    total_flagged = int(flagged.sum())

    result = {
        "injected": int(truth.sum()),
        "caught": caught,
        "missed": missed,
        "false_alarms": false_alarms,
        "flagged": total_flagged,
        # Recall: share of real problems we caught. Precision: share of our flags that were real.
        "recall": caught / truth.sum() if truth.sum() else float("nan"),
        "precision": caught / total_flagged if total_flagged else float("nan"),
        "by_type": None,
    }
    if "anomaly_type" in scored.columns and truth.any():
        planted = scored[truth]
        by_type = planted.groupby("anomaly_type")["flagged"].agg(caught="sum", injected="size")
        by_type["missed"] = by_type["injected"] - by_type["caught"]
        by_type["catch_rate"] = by_type["caught"] / by_type["injected"]
        result["by_type"] = by_type.reset_index()
    return result
