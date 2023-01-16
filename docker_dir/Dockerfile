FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Prague

RUN apt update
RUN apt install -y python3-pip

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

RUN pip3 install ogb

RUN mkdir /work
WORKDIR /work


