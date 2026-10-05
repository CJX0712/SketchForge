FROM python:3.13-slim

WORKDIR /app
COPY requirements.lock.txt pyproject.toml ./
RUN python -m pip install --no-cache-dir -r requirements.lock.txt
COPY . .
RUN python -m pip install --no-cache-dir --no-deps -e .

ENV SKETCHFORGE_BACKEND=tier1
CMD ["python", "examples/run_demo.py"]
