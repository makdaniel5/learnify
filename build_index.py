import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DATA_PATH = "./data/"
HUMAN_DB_PATH = "./db/human"
REAL_DB_PATH = "./db/real"


def build_vectorstore(module_type: str):
    if module_type == "human":
        chroma_path = HUMAN_DB_PATH
        glob_pattern = "human_*.md"
    elif module_type == "real":
        chroma_path = REAL_DB_PATH
        glob_pattern = "real_*.md"
    else:
        raise ValueError("Ismeretlen modul típus!")

    print(f"[{module_type.upper()}] Adatok betöltése...")
    loader = DirectoryLoader(
        DATA_PATH,
        glob=glob_pattern,
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()

    if not documents:
        print(f"FIGYELEM: Nem található fájl a(z) {glob_pattern} mintára a data/ mappában!")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"[{module_type.upper()}] {len(chunks)} szövegrészlet (chunk) létrehozva.")

    import stat

    def remove_readonly(func, path, excinfo):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    if os.path.exists(chroma_path):
        try:
            shutil.rmtree(chroma_path, onexc=remove_readonly)
        except Exception:
            for root, dirs, files in os.walk(chroma_path):
                for f in files:
                    try:
                        os.chmod(os.path.join(root, f), stat.S_IWRITE)
                        os.remove(os.path.join(root, f))
                    except Exception:
                        pass

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_path
    )
    print(f"[{module_type.upper()}] Vektoros adatbázis sikeresen elmentve ide: {chroma_path}\n")


if __name__ == "__main__":
    build_vectorstore("human")
    build_vectorstore("real")