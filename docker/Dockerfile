# Sandbox image.
    # Built automatically by execution_env.prepare_sandbox(); no manual step.
# pandas/numpy versions are passed in so they match the app (the DataFrame is sent as a pickle).
FROM python:3.12-slim
 
ARG PANDAS_VERSION
ARG NUMPY_VERSION
LABEL pandas=$PANDAS_VERSION
RUN pip install --no-cache-dir pandas==$PANDAS_VERSION numpy==$NUMPY_VERSION
 
# The worker script is bind-mounted read-only at run time.
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /tmp
USER 65534:65534