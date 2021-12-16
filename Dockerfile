FROM ubuntu:20.04

RUN apt update && apt install -y --no-install-recommends \
      wget \
      curl \
      python3-pip \
      nginx \
      ca-certificates \
      unzip && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install \
      numpy==1.21.4 \
      scipy==1.7.1 \
      flask==2.0.2 \
      gevent==21.12.0 \
      gunicorn==20.1.0 \
      onnxruntime==1.10.0 && \
    rm -rf /root/.cache

RUN mkdir -p /opt/models

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering
#  our standard output stream, which means that logs can be delivered to the
# user quickly. PYTHONDONTWRITEBYTECODE keeps Python from writing the .pyc
# files which are unnecessary in this case. We also update PATH so that the
# train and serve programs are found when the container is invoked.
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY src /opt/program
RUN chmod +x /opt/program/serve
WORKDIR /opt/program

ENTRYPOINT ["serve"]
