FROM python:3.9
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN python -c "import nltk; nltk.download('stopwords')"
COPY . /app
ENV PORT=$PORT
run echo $PORT
CMD gunicorn --bind 0.0.0.0:$PORT app:app
