FROM python:3.13-slim

WORKDIR /app

# System deps for llama-cpp-python and psycopg
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
