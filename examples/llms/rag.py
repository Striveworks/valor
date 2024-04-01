import os

from tqdm import tqdm
from txtai import Embeddings
from txtai.pipeline import Textractor

docs_path = "/Users/eric/repos/vimbook/docs"
embeddings_path = "./vimbook-embeddings"


def load_embeddings() -> Embeddings:
    if os.path.exists(embeddings_path):
        return Embeddings.load(path=embeddings_path)
    textractor = Textractor(paragraphs=True)

    def stream():
        filepaths = [
            os.path.join(root, f)
            for root, _, files in os.walk(docs_path)
            for f in files
            if f.endswith(".md")
        ]
        for fpath in tqdm(filepaths):
            for paragraph in textractor(fpath):
                yield paragraph

    embeddings = Embeddings(content=True)
    embeddings.index(stream())
    embeddings.save(embeddings_path)
    return embeddings


embeddings = load_embeddings()

for t in ["heart disease is caused by", "anxiety disorder"]:
    print(embeddings.search(t))
