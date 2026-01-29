FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./

RUN pip install uv && \
    uv sync --frozen

COPY main.py database.py database_schema.sql report_generator.py ./

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "main.py", "--server.address", "0.0.0.0", "--server.port", "8501"]