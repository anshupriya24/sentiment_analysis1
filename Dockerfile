FROM python:3.9.6-slim-buster
COPY . /usr/app
EXPOSE 5000
WORKDIR /usr/app
RUN pip install -r requirements.txt
CMD ["python", "sentiment.py"]