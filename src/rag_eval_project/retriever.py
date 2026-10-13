import os
import re
import glob
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

DATA_DIR = 'data'
DB_DIR = 'chroma_db'


def load_transcripts():

    docs = []

    for path in glob.glob(pathname= f"{DATA_DIR}/*.vtt"):
        lines = []

        for line in open(path):
            line = line.strip()
            if not line or line == "WEBVTT" or '-->' in line:
                continue

            lines.append(line)

        text = " ".join(lines)

        # now get the session number

        session = re.search(pattern=r"Session[ _]*(\d+)",string=path).group(1)

        docs.append(Document(page_content=text,metadata = {'session_id':session}))

    return docs

def load_store():
    embedding_model = OpenAIEmbeddings(model = 'text-embedding-3-small')

    if os.path.exists(path = DB_DIR):
        return Chroma(embedding_function=embedding_model,persist_directory=DB_DIR)

    docs = load_transcripts()

    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 150)

    chunks = splitter.split_documents(documents = docs)

    vector_store = Chroma.from_documents(documents = chunks,embedding=embedding_model,persist_directory=DB_DIR)

    return vector_store

def build_retriever():
    vector_store = load_store()
    retriever = vector_store.as_retriever(search_type = 'similarity',search_kwargs = {'k':5})
    return retriever


if __name__=='__main__':
    retriever = build_retriever()

    results = retriever.invoke('what is the regression testing')

    for r in results:
        print(f"[Session {r.metadata['session_id']}] {r.page_content[:150]}...\n")
