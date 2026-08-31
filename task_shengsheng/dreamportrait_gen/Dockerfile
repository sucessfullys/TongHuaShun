FROM hub-dev.hexin.cn:9544/hithink-gallery/dreamshop-image-label:v2

LABEL "description"="DreamFace HTTP Service"

WORKDIR /python-app
COPY . /python-app

ENV DIFFUSERS_MODEL_BASE_PATH=/mnt/data/image-edit/datasets/shensheng/models
ENV DIFFSYNTH_SKIP_DOWNLOAD=true

COPY ./readiness.sh /opt/
RUN chmod 777 /opt/readiness.sh

ENTRYPOINT ["/root/miniforge3/bin/python3.10", "start_server.py"]
CMD ["--port=9001", "--debug=False"]
