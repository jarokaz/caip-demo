
FROM gcr.io/deeplearning-platform-release/base-cpu
RUN python -m pip install -U fire gcsfs
WORKDIR /app
COPY train.py .

ENTRYPOINT ["python", "train.py"]
