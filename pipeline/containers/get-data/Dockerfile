FROM python:3.7
ADD get_data.py ./
ADD requirements.txt ./
RUN pip install -r requirements.txt
ENTRYPOINT ["python", "./get_data.py"]