# for efficiency, import argparse only first, process args, and only import heavy libs if needed

import argparse

parser = argparse.ArgumentParser(description='Embed images in a directory recursively, into a database using chromadb and siglip')

import os
def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"{path} does not exist or is not a directory")

parser.add_argument('dir', type=dir_path, help='Directory to embed images from')

#optionally choose a different database directory
parser.add_argument('--db', type=str, help='Path to database directory', default="~/.cache/img_embed")

args = parser.parse_args()



from PIL import Image
import os.path as path
import chromadb
from transformers import (
    AutoProcessor,
    AutoModel
)
import torch
import hashlib

if not path.exists("~/.cache/img_embed") and args.db == "~/.cache/img_embed/embeds.db":
    os.makedirs(os.path.expanduser("~/.cache/img_embed"))
    os.makedirs(os.path.expanduser("~/.cache/img_embed/") + path.basename(args.dir))

class EmbedImgDir:
    def __init__(self, processing_dir) -> None:
        # init model and processor
        self.model_name = "google/siglip-so400m-patch14-384"
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        # init db
        if args.db == "~/.cache/img_embed/embeds.db":
            self.db_path = os.path.expanduser("~/.cache/img_embed/") + path.basename(processing_dir) + "/embeds.db"
        else:
            self.db_path = args.db + "/embeds.db"
        self.client = chromadb.PersistentClient(self.db_path)
        # if the collection does not exist, create it
        self.collection = self.client.get_or_create_collection("embeds", metadata={"model": self.model_name, "dir": processing_dir})


    @torch.inference_mode
    def embed_fn(self, img: Image.Image):
        inputs = self.processor(images=[img], return_tensors="pt")
        return (self.model.get_image_features(**inputs)).squeeze().tolist()
        
        
    def processing(self):
        for root, _, files in os.walk(args.dir):
            for file in files:
                img_path = path.join(root, file)
                # get hash for use as ID
                img_id = hashlib.md5(img_path.encode()).hexdigest()
                img = Image.open(img_path)
                # get embedding
                embedding = self.embed_fn(img)
                self.collection.add(img_id, embeddings=embedding, metadata={"path": img_path})
                # print
                print(f"Added {img_path} to database with ID {img_id}")


if __name__ == "__main__":
    embedder = EmbedImgDir(args.dir)
    embedder.processing()


