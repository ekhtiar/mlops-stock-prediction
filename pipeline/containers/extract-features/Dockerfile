FROM python:3.7
ADD extract_features.py ./
ADD requirements.txt ./
RUN pip install -r requirements.txt
ENTRYPOINT ["python", "./extract_features.py"]