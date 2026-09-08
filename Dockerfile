# Slim, CPU-only image for the DoorDash ETA demo (no torch needed at serve time)
FROM python:3.10-slim

WORKDIR /app

# Install only the lightweight serving deps
COPY app/requirements.txt ./app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt

# App code, feature engineering, and the trained quantile models
COPY src/ ./src/
COPY models/quantile_p10.joblib models/quantile_p50.joblib models/quantile_p90.joblib ./models/
COPY models/feature_defaults.json ./models/
COPY app/ ./app/

WORKDIR /app/app
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
