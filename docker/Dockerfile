FROM python:3.12-slim

WORKDIR /app

ARG MODEL_URL="https://huggingface.co/Rahul040202/patchcore-transistor-inspection/resolve/main/patchcore_model.pt"
ARG MODEL_PATH="/app/app/weights/patchcore_model.pt"

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

RUN mkdir -p "$(dirname "$MODEL_PATH")" \
    && if [ ! -f "$MODEL_PATH" ]; then \
        curl -L --fail --retry 3 --output "$MODEL_PATH" "$MODEL_URL"; \
    fi

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
