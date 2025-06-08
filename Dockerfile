FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    python3-pip \
    graphviz \
    graphviz-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio

RUN pip install --no-cache-dir \
    torch-geometric \
    pyg-lib \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv

RUN pip install --no-cache-dir \
    pytorch-lightning \
    fastapi \
    uvicorn \
    redis \
    faiss-cpu \
    transformers \
    pandas \
    scikit-learn \
    praw \
    python-dotenv \
    tqdm \
    networkx \
    matplotlib \
    plotly \
    dash

WORKDIR /app

COPY . /app/

CMD ["python"] 