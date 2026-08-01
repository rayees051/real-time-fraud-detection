# api/app.py
from fastapi import FastAPI
from pydantic import BaseModel
from kafka import KafkaProducer
import json, time
from datetime import datetime

app = FastAPI(title="Fraud Ingest API")

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

class Transaction(BaseModel):
    customer_id: str
    transaction_id: str
    amount: float
    timestamp: str
    merchant_id: str
    device_info: str
    city: str
    lat: float
    lon: float

@app.post("/transaction")
def ingest(tx: Transaction):
    data = tx.dict()

    # Convert normal date/time into Unix timestamp
    dt = datetime.strptime(tx.timestamp, "%Y-%m-%d %H:%M:%S")
    data["timestamp"] = int(dt.timestamp())

    # Publish transaction to Kafka
    producer.send("incoming_transactions", value=data)
    producer.flush()

    return {
        "status": "success",
        "message": "sent to incoming_transactions",
        "data": data
    }
