FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Serving libraries
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Exact numpy/pandas/scikit-learn/joblib versions the model was trained with
COPY deployment/requirements-model.txt ./requirements-model.txt
RUN pip install --no-cache-dir -r requirements-model.txt

COPY api/ ./api/
COPY deployment/ ./deployment/
COPY config.yaml .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
