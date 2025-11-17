FROM python:3.10-slim

# create app dir
WORKDIR /app

# install system deps
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# copy reqs
COPY requirements.txt .

# install python deps
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# copy app code
COPY . .

# download model from Google Drive at build-time
RUN mkdir -p models && \
    apt-get update && apt-get install -y wget curl && \
    curl -L "https://drive.google.com/uc?export=download&id=1CcG7idf0yIbj697E7_JoBeP_BNJ6_KIi" -o models/model.pth

# expose
EXPOSE 10000

# start
CMD ["gunicorn", "server:app", "--bind", "0.0.0.0:10000"]
