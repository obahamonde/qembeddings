import tempfile
from typing import Literal, TypeAlias, Union, Type

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from langchain.document_loaders.excel import UnstructuredExcelLoader
from langchain.document_loaders.pdf import UnstructuredPDFLoader
from langchain.document_loaders.powerpoint import UnstructuredPowerPointLoader
from langchain.document_loaders.text import TextLoader
from langchain.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain.document_loaders.json_loader import JSONLoader
from langchain.document_loaders.markdown import (
    UnstructuredMarkdownLoader as MarkdownLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

Document: TypeAlias = Union[
    Type[UnstructuredWordDocumentLoader],
    Type[UnstructuredPDFLoader],
    Type[UnstructuredExcelLoader],
    Type[UnstructuredPowerPointLoader],
    Type[TextLoader],
    Type[JSONLoader],
    Type[MarkdownLoader],
]
ContentType: TypeAlias = Literal[
    "word",
    "pdf",
    "excel",
    "powerpoint",
    "text",
    "json",
    "jsonl",
    "yaml",
    "csv",
    "md",
    "markdown",
]

MAPPING: dict[ContentType, Document] = {
    "word": UnstructuredWordDocumentLoader,
    "pdf": UnstructuredPDFLoader,
    "excel": UnstructuredExcelLoader,
    "powerpoint": UnstructuredPowerPointLoader,
    "text": TextLoader,
    "json": JSONLoader,
    "jsonl": JSONLoader,
    "yaml": JSONLoader,
    "csv": JSONLoader,
    "md": MarkdownLoader,
    "markdown": MarkdownLoader,
}


def check_content_type(file: UploadFile) -> ContentType:
    if file.content_type is None and file.filename is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No content type or filename provided",
        )
    if file.content_type or (file.content_type and file.filename):
        if "word" in file.content_type:
            return "word"
        if "pdf" in file.content_type:
            return "pdf"
        if "excel" in file.content_type:
            return "excel"
        if "powerpoint" in file.content_type:
            return "powerpoint"
        if "json" in file.content_type:
            return "json"
        if "yaml" in file.content_type:
            return "yaml"
        if "csv" in file.content_type:
            return "csv"
        if "jsonl" in file.content_type:
            return "jsonl"
        if "md" in file.content_type:
            return "markdown"
        if "markdown" in file.content_type:
            return "markdown"
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported content type",
        )
    if file.filename:
        if "doc" in file.filename:
            return "word"
        if "pdf" in file.filename:
            return "pdf"
        if "xls" in file.filename:
            return "excel"
        if "ppt" in file.filename:
            return "powerpoint"
        if "txt" in file.filename:
            return "text"
        if "json" in file.filename:
            return "json"
        if "jsonl" in file.filename:
            return "jsonl"
        if "yaml" in file.filename:
            return "yaml"
        if "csv" in file.filename:
            return "csv"
        if "md" in file.filename:
            return "markdown"
        if "markdown" in file.filename:
            return "markdown"
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file extension",
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Unreachable code"
        )


app = APIRouter()


@app.post("/documents")
async def load_document(file: UploadFile = File(...)):
    content_type = check_content_type(file)
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(await file.read())
        temp.seek(0)
        documents = MAPPING[content_type](temp.name).load_and_split()
        splitter = RecursiveCharacterTextSplitter()
        return [doc.page_content for doc in splitter.split_documents(documents)]
