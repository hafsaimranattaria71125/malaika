# ── Base Image ─────────────────────────────────────────────
FROM python:3.11-slim

# ── Install dependencies ───────────────────────────────────
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project files ─────────────────────────────────────
COPY main.py .
COPY model.py .
COPY schemas.py .
COPY train.py .

# Copy model artifacts
COPY logistic_model.pkl .
COPY scaler.pkl .
COPY stats.pkl .

# Copy dataset
COPY diabetes.csv .

# ── Expose FastAPI port ────────────────────────────────────
EXPOSE 7860

# ── Run app ────────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
