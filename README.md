# larex-action-pagexml-ner

Reference LAREX postprocessing Action that converts PAGE XML into reading-order-aware text and optional named-entity exports. Its files are stored as one durable entry in the project's **Outputs** panel.

## Output

Every successful run produces a flat collection:

| File | Association | Description |
| --- | --- | --- |
| `0001-<page>.txt` | Page | Normalized text in PAGE reading order. |
| `0001-<page>.entities.json` | Page | Entity text, label, character offsets, and optional knowledge-base id. |
| `document.txt` | Project | All successfully processed page texts separated by form feeds. |
| `entities.csv` | Project | Searchable project-wide entity index with page and document offsets. |
| `manifest.json` | Project | Processor/configuration provenance, page statistics, warnings, sizes, and SHA-256 checksums. |

The complete collection is submitted in one final callback. This makes publication atomic: a failed or cancelled run leaves no partial output. Individual page files still carry their LAREX `pageId` in the result manifest.

## Text extraction

- Supports namespace-qualified PAGE XML versions without binding to one schema date.
- Follows nested PAGE `ReadingOrder` groups and appends unreferenced text regions in document order.
- Selects the lowest-index `TextEquiv` and falls back to region-level text when lines have no text.
- Optionally preserves line breaks, joins likely line-end hyphenation, and applies NFC or NFKC normalization.
- Parses XML with DTD loading, networking, and entity resolution disabled.

Dehyphenation is intentionally conservative: a trailing hyphen is removed only when the next non-empty line starts with a lowercase character.

## Named-entity recognition

The container includes spaCy `en_core_web_sm` 3.8.0 and enables it in the supplied Action definition. The Docker build loads the model once to validate the artifact. At runtime the processor loads and warms the configured preload model in the background during application startup, so the first Action run does not pay that cost. Set `enableNer` to `false` to use the processor as a deterministic plain-text exporter.

`nerModel` may name another spaCy package or an absolute model path baked into or mounted in the container. The preloaded model and any alternative models are cached for the process lifetime. Alternative `nerModel` values are loaded lazily. `entityLabels` is a comma-separated allowlist such as `PERSON,ORG,GPE,LOC,DATE`; an empty value includes every label emitted by the model.

The small English model is suitable as an accessible reference, not as a domain-independent accuracy baseline. Production deployments should pin and validate a model appropriate for their language and documents.

## Action definition

Register [action/pagexml-text-ner-export.yaml](action/pagexml-text-ner-export.yaml) in LAREX and replace the example endpoints:

```yaml
endpoint:
  url: https://pagexml-ner.example.org/dispatch
  healthUrl: https://pagexml-ner.example.org/health
  preflightUrl: https://pagexml-ner.example.org/preflight
  auth:
    type: hmac
    secretRef: pagexml-text-ner-export-v1
```

Create the endpoint secret under **Admin → Actions → Endpoint Secrets**, then pass the one-time revealed value to the processor as `LAREX_DISPATCH_HMAC_SECRET`.

This processor requires a LAREX server advertising `capabilities.customFileResults` and `larex-action-sdk` `>=0.10.1,<0.11`.

## Configuration

Action parameters are declared in the YAML definition and snapshotted into `manifest.json`:

| Parameter | Default | Description |
| --- | --- | --- |
| `preserveLineBreaks` | `true` | Keep PAGE text-line boundaries. |
| `dehyphenate` | `true` | Join conservative line-end hyphenation candidates. |
| `unicodeNormalization` | `NFC` | `none`, `NFC`, or `NFKC`. |
| `enableNer` | `true` | Enable spaCy NER. |
| `nerModel` | `en_core_web_sm` | Installed spaCy package or model path. |
| `entityLabels` | `PERSON,ORG,GPE,LOC,DATE` | Comma-separated label allowlist; empty means all. |
| `continueOnInvalidXml` | `false` | Skip invalid pages and record warnings. |

Processor environment:

| Variable | Default | Description |
| --- | --- | --- |
| `LAREX_DISPATCH_HMAC_SECRET` | required | Shared dispatch HMAC secret. |
| `LAREX_PROCESSOR_ID` | `pagexml-text-ner-export` | Expected processor id. |
| `LAREX_ALLOWED_CALLBACK_ORIGINS` | SDK default | Optional comma-separated callback origins. |
| `LAREX_ACTION_ROUTE_PREFIXES` | empty | Optional route prefixes exposed by the SDK app. |
| `LAREX_PRELOAD_NER_MODEL` | `en_core_web_sm` | Model loaded and warmed during startup; empty disables preloading. |
| `LAREX_MAX_CONCURRENT_RUNS` | `1` | Runs admitted concurrently by one instance. |
| `LAREX_MAX_XML_BYTES` | `52428800` | Per-page PAGE XML size guard. |

`/health` is the liveness endpoint. `/preflight` is the authenticated LAREX
configuration check; it verifies the shared HMAC secret and processor id and
reports protocol and result capabilities. `/ready` returns `503` while the preload
model is loading, if loading failed, or while all configured run slots are
occupied. A successful readiness response includes the warmed model name.
Prefixed routes follow `LAREX_ACTION_ROUTE_PREFIXES` as well.

## Run locally

Install the released SDK and development dependencies with uv:

```bash
uv sync --extra ner --extra test
export LAREX_DISPATCH_HMAC_SECRET='dev-secret'
uv run uvicorn larex_action_pagexml_ner.main:app --host 0.0.0.0 --port 9000
```

Run tests:

```bash
uv run pytest
uv run ruff check .
```

## Docker

```bash
docker build -t larex-action-pagexml-ner .
docker run --rm -p 9000:9000 \
  -e LAREX_DISPATCH_HMAC_SECRET='secret-from-larex' \
  -e LAREX_ALLOWED_CALLBACK_ORIGINS='https://larex.example.org' \
  larex-action-pagexml-ner
```

Tagged images are configured for publication to `ghcr.io/ocr4all/larex-action-pagexml-ner`.
