# Usa una immagine Python stabile
FROM python:3.10-slim

# Evita prompt interattivi
ENV DEBIAN_FRONTEND=noninteractive

# Lavora nella cartella /app
WORKDIR /app

# Installa dipendenze di sistema per OpenCV + pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libjpeg62-turbo \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

# Copia i requirements
COPY requirements.txt .

# Installa le librerie Python
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Copia tutto il codice nell'immagine
COPY . .

# Crea la cartella results (necessaria per salvataggi)
RUN mkdir -p /app/results

# Espone la porta usata da Render
EXPOSE 10000

# Comando di avvio con gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "server:app"]
