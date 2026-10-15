FROM python:3.12-slim
WORKDIR /app
COPY requirements.lock ./requirements.lock
RUN pip install --no-cache-dir -r requirements.lock
COPY app ./app
COPY static ./static
COPY data/dataset_report.json ./data/dataset_report.json
RUN useradd --create-home --uid 10001 researcher && mkdir /app/runtime && chown -R researcher:researcher /app
USER researcher
EXPOSE 8765
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8765"]
