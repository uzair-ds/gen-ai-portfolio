from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from config import VECTOR_STORE_PATH


def load_vector_store():
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
