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

# maybe if they want the model quantized to save memory
parser.add_argument('--quantized', action='store_true', help='Quantize the model to save memory')

args = parser.parse_args()



from PIL import Image
import os.path as path
import chromadb
from transformers import (
    AutoProcessor,
    AutoModel
)

if not path.exists("~/.cache/img_embed"):
    os.makedirs("~/.cache/img_embed")
    os.makedirs("~/.cache/img_embed/" + path.basename(args.dir))
chroma_client = chromadb.PersistentClient("~/.cache")