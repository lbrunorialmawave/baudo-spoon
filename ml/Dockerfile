FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy into /app/ml/ so the package is importable as `ml.*`
COPY . /app/ml/

# Artifacts (models, plots, JSON results) are written here.
# Mount a volume to persist them outside the container.
VOLUME ["/app/artifacts"]

ENTRYPOINT ["python", "-m", "ml.run_pipeline"]
