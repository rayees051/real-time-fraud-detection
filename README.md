# Real-Time Fraud Detection System

Built an end-to-end real-time fraud detection pipeline using Machine Learning, Kafka streaming, FastAPI, and Streamlit dashboarding.

A Machine Learning–based real-time fraud detection system that processes live financial transactions, performs behavioural analysis, and predicts whether a transaction is SAFE or FRAUD in real time.

---

#  Features

- Real-time transaction streaming using Apache Kafka
- Machine Learning fraud prediction
- Behavioural feature engineering
- Live fraud monitoring dashboard using Streamlit
- FastAPI-based transaction ingestion API
- Real-time scoring pipeline
- Fraud probability prediction
- Customer behavioural analysis

---

#  Tech Stack

- Python
- Apache Kafka
- FastAPI
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Docker

---

#  Machine Learning Features

The model uses behavioural and statistical features such as:

- Transaction amount analysis
- Amount log transformation
- Customer transaction frequency
- Average customer spending
- Device change detection
- Location change detection
- Risk score calculation
- Night-time transaction analysis

---

#  System Architecture

Transaction API → Kafka Producer → Enrichment Consumer → ML Scoring Consumer → Predictions Storage → Streamlit Dashboard

---

# Dashboard Features

- Live transaction monitoring
- Fraud alert visualization
- Real-time predictions
- Customer transaction tracking
- Fraud probability display

---

#  How to Run

## 1. Start Docker Services

```bash
docker compose up -d
```

## 2. Start Enrichment Consumer

```bash
python streaming/consumer_enrich.py
```

## 3. Start Scoring Consumer

```bash
python streaming/consumer_score.py
```

## 4. Start FastAPI Server

```bash
uvicorn api.app:app --reload
```

## 5. Start Streamlit Dashboard

```bash
streamlit run ui/st.py
```

---

#  Future Enhancements

- Trust recovery mechanism
- Risk score decay
- Online learning system
- SMS / Email fraud alerts
- Advanced dashboard analytics
- Adaptive fraud scoring

---

#  Author

Rayees Akbar