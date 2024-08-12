import os
import requests
import yaml
import pathlib
import glob
import time
from langchain_text_splitters import MarkdownTextSplitter

from dotenv import load_dotenv

def embed(content: str):
   tries = 3
   timeout = 1

   for _i in range(tries):
    try:
        resp = requests.post('https://curated.aleph.cloud/vm/ee1b2a8e5bd645447739d8b234ef495c9a2b4d0b98317d510a3ccf822808ebe5/embedding', json={'content': content})
        return resp.json()["embedding"]
    except:
        time.sleep(timeout)


class _Config:
    debug: bool
    ai_config: dict

    documents: str

    def __init__(self):
        load_dotenv()

        self.debug = os.getenv("DEBUG", "False") == "True"

        ai_config_path = os.getenv("GENERAL_CONFIG_PATH", "config/general.yaml")
        with open(ai_config_path) as ai_config_file:
            self.ai_config = yaml.safe_load(ai_config_file)

        directory = pathlib.Path().resolve()
        documents = []
        for filename in glob.iglob(f'{directory}/docs/**/*.md', recursive=True):
            with open(filename) as file:
                content = file.read()
                document_chunks = MarkdownTextSplitter(chunk_size=500, chunk_overlap=100).create_documents([content])
                for chunk in document_chunks:
                    embedding_vector = embed(chunk.page_content)
                    documents.append({'content': chunk.page_content, 'vector': embedding_vector})
        self.documents = documents
env = _Config()
