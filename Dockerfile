# Usa Python 3.10 (compatibile con torch 2.1)
FROM python:3.10-slim

# Variabili d'ambiente
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependencies di sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Crea directory app
WORKDIR /app

# Copia requirements
COPY requirements.txt .

# Installa dipendenze pip (incluso torch CPU)
RUN pip install --no-cache-dir -r requirements.txt

# Copia tutto il progetto
COPY . .

# Espone la porta di Render
ENV PORT=10000
EXPOSE 10000

# Comando di avvio
CMD ["gunicorn", "server:app", "--bind", "0.0.0.0:10000", "--timeout", "300"]
