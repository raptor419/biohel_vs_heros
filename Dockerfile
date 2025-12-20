FROM --platform=linux/amd64 ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y g++ make flex && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY BioHEL-cuda ./BioHEL-cuda
WORKDIR /app/BioHEL-cuda
RUN touch .depend && make clean && make
