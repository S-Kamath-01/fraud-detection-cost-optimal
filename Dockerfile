# Production image for the fraud detection API.
#
# Uses requirements-api.txt, NOT requirements.txt - the latter is a full
# pip freeze of the local dev environment (Jupyter, notebooks, etc.) and
# includes pywinpty, a Windows-only package that cannot build on Linux.
# The API doesn't need Jupyter at all, so this image only installs what
# src/main.py and the modules it imports actually require.
#
# Uses psycopg2-binary rather than compiling psycopg2 from source. The
# more "correct" production choice is compiling against the system's own
# libpq/OpenSSL so security patches arrive via normal apt updates - but
# that requires a working gcc/libpq-dev toolchain in the image, which is
# extra build surface area for a portfolio project's benefit. Documented
# here as a deliberate, known tradeoff, not an oversight.

FROM python:3.12-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Only what's needed at runtime - not data/, notebooks, or .git.
# checkpoints/ contains only the 4 committed deployment artifacts
# (xgb_model.json, feature_columns.json, threshold_config.json,
# drift_reference.csv). The inference service never needs the raw or
# processed dataset - drift_reference.csv is a small sampled stand-in
# for train.csv, generated once by train.py.
COPY src/ src/
COPY checkpoints/ checkpoints/

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]