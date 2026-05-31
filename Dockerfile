FROM node:18-alpine AS ui-build

WORKDIR /ui
COPY ui/package*.json ./
RUN npm install
COPY ui/ ./
ARG VITE_API_BASE=
ENV VITE_API_BASE=$VITE_API_BASE
RUN npm run build

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY artifacts/ artifacts/
COPY --from=ui-build /ui/dist /app/ui/dist

ENV MODEL_PATH=artifacts/model.pkl
ENV METRICS_PATH=artifacts/metrics.json
ENV UI_DIST=/app/ui/dist

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

