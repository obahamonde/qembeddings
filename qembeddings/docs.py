import tempfile
from typing import Literal, TypeAlias, Union

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from langchain.document_loaders.excel import UnstructuredExcelLoader
from langchain.document_loaders.pdf import UnstructuredPDFLoader
from langchain.document_loaders.powerpoint import UnstructuredPowerPointLoader
from langchain.document_loaders.text import TextLoader
from langchain.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

Document: TypeAlias = Union[
    UnstructuredWordDocumentLoader,
    UnstructuredPDFLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    TextLoader,
]
ContentType: TypeAlias = Literal["word", "pdf", "excel", "powerpoint", "text"]

MAPPING = {
    "word": UnstructuredWordDocumentLoader,
    "pdf": UnstructuredPDFLoader,
    "excel": UnstructuredExcelLoader,
    "powerpoint": UnstructuredPowerPointLoader,
    "text": TextLoader,
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
        if "text" in file.content_type:
            return "text"
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
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid content type"
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
