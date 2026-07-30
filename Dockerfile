FROM python:3.12-slim

# libmagic1: native lib required by python-magic (used by DataTrove's WarcReader)
# curl: used by k8s Jobs to fetch WET files directly from Common Crawl in-cluster
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-pipeline.txt .
RUN pip install --no-cache-dir -r requirements-pipeline.txt

COPY pipeline/ ./pipeline/

ENTRYPOINT ["python"]
