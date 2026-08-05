FROM python:3.12-slim

WORKDIR /app

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ .

# ml module is required at runtime (imported by api.src.routers.auction)
COPY ml/ /app/ml/

# scraper module is required at runtime (imported by admin_scrape, matchday routers)
COPY scraper/ /app/scraper/

# Baked into the image so POST /admin/scrape/quotazioni finds the listoni on
# platforms without a host filesystem to bind-mount (e.g. Render) — the
# docker-compose.yml volume mount only helps local dev, where it shadows this.
COPY quotazioni/ /app/quotazioni/

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
