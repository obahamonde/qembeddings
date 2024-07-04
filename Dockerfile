FROM python:3.11
WORKDIR /app
COPY . /app
EXPOSE 4500
RUN pip install -r requirements.txt
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","4500","--reload"]
