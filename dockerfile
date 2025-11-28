FROM python:3.10-slim


RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    git \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app


COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


COPY app.py /app/app.py
COPY mnist_cnn.pth /app/mnist_cnn.pth
COPY lstm_model.h5 /app/lstm_model.h5
COPY lstm_scaler.pkl /app/lstm_scaler.pkl


EXPOSE 8501
RUN mkdir -p /app/.streamlit


RUN bash -c 'echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = 8501\n\
" > /app/.streamlit/secrets.toml'


CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
