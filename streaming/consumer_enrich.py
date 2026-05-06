# streaming/consumer_enrich.py
import json, os, sqlite3, math
from kafka import KafkaConsumer, KafkaProducer
from datetime import datetime

DB_PATH = "data/customer_state.db"

def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS customer_state (
        customer_id TEXT PRIMARY KEY,
        txn_count INTEGER,
        cum_amount REAL,
        last_device TEXT,
        last_city TEXT,
        last_timestamp INTEGER
    )
    """)
    conn.commit()
    conn.close()

def get_state(conn, cust):
    c = conn.cursor()
    c.execute("SELECT txn_count, cum_amount, last_device, last_city, last_timestamp FROM customer_state WHERE customer_id = ?", (cust,))
    r = c.fetchone()
    if r:
        return {"txn_count": r[0], "cum_amount": r[1],
                "last_device": r[2], "last_city": r[3], "last_timestamp": r[4]}
    else:
        return {"txn_count": 0, "cum_amount": 0.0, "last_device": None, "last_city": None, "last_timestamp": None}

def update_state(conn, cust, txn_count, cum_amount, device, city, ts):
    c = conn.cursor()
    c.execute("""
      INSERT INTO customer_state(customer_id, txn_count, cum_amount, last_device, last_city, last_timestamp)
      VALUES(?,?,?,?,?,?)
      ON CONFLICT(customer_id) DO UPDATE SET
        txn_count=excluded.txn_count,
        cum_amount=excluded.cum_amount,
        last_device=excluded.last_device,
        last_city=excluded.last_city,
        last_timestamp=excluded.last_timestamp
    """, (cust, txn_count, cum_amount, device, city, ts))
    conn.commit()

# Kafka
consumer = KafkaConsumer(
    'incoming_transactions',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='latest',
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

print("Initializing DB...")
init_db()
conn = sqlite3.connect(DB_PATH, check_same_thread=False)

print("Enrichment consumer started — listening on incoming_transactions")
for msg in consumer:
    rec = msg.value
    cust = rec.get("customer_id")
    amount = float(rec.get("amount", 0.0))
    ts = int(rec.get("timestamp", 0))
    device = rec.get("device_info", "")
    city = rec.get("city", "")

    # load previous state
    state = get_state(conn, cust)
    prev_count = int(state["txn_count"])
    prev_cum = float(state["cum_amount"])
    prev_device = state["last_device"]
    prev_city = state["last_city"]

    # compute features BEFORE updating state (so they reflect history)
    cust_txn_count = prev_count
    cust_cum_mean = (prev_cum / prev_count) if prev_count > 0 else 0.0
    amt_to_cum_mean = (amount / cust_cum_mean) if cust_cum_mean > 0 else 1.0
    is_new_device = 1 if (prev_device is None or device != prev_device) else 0
    # new_city_flag: 1 if different from last city (history). First txn -> 0
    new_city_flag = 1 if (prev_city is not None and city != prev_city) else 0

    # time features
    hour = int((ts // 3600) % 24)
    is_night = 1 if (hour < 6 or hour > 22) else 0
    amount_log = math.log1p(amount)

    enriched = dict(rec)  # copy raw fields
    enriched.update({
        "cust_txn_count": cust_txn_count,
        "cust_cum_mean": cust_cum_mean,
        "amt_to_cum_mean": amt_to_cum_mean,
        "is_new_device": is_new_device,
        "new_city_flag": new_city_flag,
        "hour": hour,
        "is_night": is_night,
        "amount_log": amount_log
    })

    # debug: show exactly what enrichment publishes (helpful while testing)
    print("➡️ SENDING TO SCORING:", {k: enriched.get(k) for k in ['customer_id','transaction_id','amount','cust_txn_count','cust_cum_mean','amt_to_cum_mean','is_new_device','new_city_flag','hour','is_night']})

    # publish to scored_transactions for scoring consumer
    producer.send("scored_transactions", enriched)
    producer.flush()

    # now update state to include this transaction
    new_count = prev_count + 1
    new_cum = prev_cum + amount
    update_state(conn, cust, new_count, new_cum, device, city, ts)
