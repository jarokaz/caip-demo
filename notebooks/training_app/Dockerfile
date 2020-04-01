
FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-1
RUN pip install -U fire
WORKDIR /app
COPY train.py .

ENTRYPOINT ["python", "train.py"]
