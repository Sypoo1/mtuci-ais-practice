FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including build tools
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

# Copy requirements
COPY pyproject.toml uv.lock ./

# Install uv and dependencies
RUN pip install uv && \
    uv sync --frozen

# Copy application files
COPY main.py database.py database_schema.sql ./

# Expose Streamlit port
EXPOSE 8501

# Run the application
CMD ["uv", "run", "streamlit", "run", "main.py", "--server.address", "0.0.0.0", "--server.port", "8501"]