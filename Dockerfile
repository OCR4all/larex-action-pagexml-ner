FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.9.17 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --frozen --no-dev --no-editable --extra ner

RUN /opt/venv/bin/python -c "import spacy; spacy.load('en_core_web_sm')('LAREX build validation.')"

FROM python:3.12-slim

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LAREX_PRELOAD_NER_MODEL=en_core_web_sm

WORKDIR /app

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin larex-action

COPY --from=builder /opt/venv /opt/venv

USER 10001:10001

EXPOSE 9000

CMD ["uvicorn", "larex_action_pagexml_ner.main:app", "--host", "0.0.0.0", "--port", "9000"]
