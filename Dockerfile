FROM python:3.11
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1
RUN apt-get update && apt-get install -y poppler-utils \
	&& apt-get install tesseract-ocr -y \
	&& apt-get install libgl1-mesa-glx -y 
WORKDIR /app
COPY . /app
EXPOSE 4500
RUN pip install -r requirements.txt
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","4500"]
