FROM python:3.12-slim

WORKDIR /workspace/project
COPY factory_sre /workspace/project/factory_sre

ENV PYTHONUNBUFFERED=1
