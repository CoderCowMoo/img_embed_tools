import torch
import os.path as path
import os
import chromadb
from PIL import Image
import time
import hashlib


class EmbedImgDir:
    def __init__(self, processing_dir, model_name, processor, model, args) -> None:
        # init model and processor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.processor = processor
        self.model = model
        self.args = args
        # init db
        if args.db == "~/.cache/img_embed/embeds.db":
            self.db_path = os.path.expanduser("~/.cache/img_embed/") + path.basename(processing_dir) + "/embeds.db"
        else:
            self.db_path = args.db + "/embeds.db"
        self.client = chromadb.PersistentClient(self.db_path)
        # if the collection does not exist, create it
        try:
            self.collection = self.client.get_or_create_collection("embeds", metadata={"model": self.model_name, "dir": processing_dir})
        except:
            self.collection = self.client.get_collection("embeds")


    @torch.inference_mode
    def embed_fn(self, img: Image.Image):
        inputs = self.processor(images=[img], return_tensors="pt")["pixel_values"].to(self.device)
        return (self.model.get_image_features(inputs)).squeeze().tolist()
        
        
    def process_dir(self):
        # time the embedding process
        start = time.time()
        for root, _, files in os.walk(self.args.dir):
            for file in files:
                img_path = path.join(root, file)
                # get hash for use as ID
                img_id = hashlib.md5(img_path.encode()).hexdigest()
                # if alreayd in db, skip
                if self.collection.get(img_id)["embeddings"]:
                    print(f"Skipping {img_path} as it is already in the database")
                    continue
                img = Image.open(img_path)
                # get embedding
                embedding = self.embed_fn(img)
                self.collection.add(img_id, embeddings=embedding, metadatas={"path": img_path})
                # print
                print(f"Added {img_path} to database with ID {img_id}")
        print(f"Finished embedding in {time.time() - start} seconds")

    def query_text(self, query: str, tokenizer, db_path: str, num_results: int = 5):
        tokenized = tokenizer([query], return_tensors="pt", padding="max_length").to(self.model.device)
        embeds = self.model.get_text_features(**tokenized).squeeze().tolist()
        results = self.collection.query(query_embeddings=embeds, n_results=num_results)
        # results returns as a dict of:
        #   "ids": [['...', '...']]
        #   "distances": [[0.0, 0.0]]
        #   "metadatas": [{"path": "..."}]
        #   "embeddings": None
        #   "documents": [[None x 5]]
        #   "uris": None
        #   "data": None
        return results
    
    def query_image(self, query: Image.Image, num_results: int = 5):
        embeds = self.embed_fn(query)
        results = self.collection.query(query_embeddings=embeds, n_results=num_results)
        return results
