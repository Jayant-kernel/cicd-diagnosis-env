# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# install deps first for better layer caching
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# copy the package and the inference script
COPY cicd_diagnosis_env/ /app/cicd_diagnosis_env/
COPY inference.py /app/inference.py

ENV PYTHONPATH=/app
EXPOSE 7860

HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

CMD ["uvicorn", "cicd_diagnosis_env.server.app:app", \
     "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
