FROM python:3.9-slim

WORKDIR /app

COPY . /app

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*
    
# Install core dependencies.
RUN apt-get update && apt-get install -y 
# libpq-dev build-essential
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


# Expose the application port
EXPOSE 8000

# Command to run FastAPI using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]