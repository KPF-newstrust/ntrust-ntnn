FROM ubuntu:18.04

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

RUN \
    apt update && \
    apt install -y python3-dev python3-pip git

RUN pip3 install pipenv

# disable interactive functions
ENV DEBIAN_FRONTEND noninteractive

# install mecab
RUN apt install -y g++ curl swig
RUN curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh > mecab.sh
RUN /bin/bash -c "source ./mecab.sh" && rm mecab.sh

RUN \
    apt install -y --no-install-recommends ca-certificates && \
    apt install -y tzdata

RUN ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime

RUN set -ex && mkdir /app

WORKDIR /app

#COPY Pipfile Pipfile
#COPY Pipfile.lock Pipfile.lock

#RUN pipenv install --deploy --system

COPY . /app

#CMD python3 app.py
