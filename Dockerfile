FROM python:3.11

WORKDIR /app
COPY . ./

RUN pip install --upgrade pip
RUN pip install -r ./deploy-google/requirements.txt

EXPOSE 5000

CMD ["sh", "-c", "cd deploy-google && python3 -m flask run --host=0.0.0.0"]