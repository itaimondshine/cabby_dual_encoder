
# Dockerfile
FROM python:3.7-stretch
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
COPY . /app_instructor
WORKDIR /app_instructor
RUN pip install -r requirements.txt
RUN python -c "import nltk;nltk.download('punkt');nltk.download('averaged_perceptron_tagger')"
ENTRYPOINT ["python"]
CMD ["app_instructor.py"]
