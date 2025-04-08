FROM m.daocloud.io/python:3.9

RUN mkdir /agent

RUN pip3 config --user set global.progress_bar off
RUN pip3 install prometheus_client

# 假设本地有两个文件 file1.txt 和 file2.txt，将它们复制到容器的 /data 文件夹中
COPY agent.py /agent/
COPY main.py /agent/