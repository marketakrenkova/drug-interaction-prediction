FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Prague

RUN apt update
RUN apt install -y python3-pip

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN mkdir /work
WORKDIR /work


