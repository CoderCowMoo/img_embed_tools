# embed_img_dir.py

A python tool to embed all images in a directory recursively.

Relies on:
- SigLIP via 🤗 Transformers
- ChromaDB for embed storage


Stores a chromadb database at $HOME\.cache\img_embeds\{DIR_NAME}\embeds.db
Updates the database if the tool is run on the same directory multiple times.

### References:
- https://stackoverflow.com/questions/20063/whats-the-best-way-to-parse-command-line-arguments
- https://docs.trychroma.com/usage-guide
- https://pytorch.org/docs/stable/generated/torch.Tensor.tolist.html
- https://github.com/codercowmoo/llm-clip (my siglip fork)
- Guaranteed I missed some,