from argparse import ArgumentParser
cli = ArgumentParser(description='Image embed tools, including embedding images in a directory and querying the database, etc.')

subparsers = cli.add_subparsers(dest='subcommand', title='subcommands', description='valid subcommands', help='additional help')


# Setup logic for subcommands
def subcommand(args=[], parent=subparsers):
    def decorator(func):
        parser = parent.add_parser(func.__name__, description=func.__doc__)
        for arg in args:
            parser.add_argument(*arg[0], **arg[1])
        parser.set_defaults(func=func)
    return decorator

def argument(*name_or_flags, **kwargs):
    """Convenience function to properly format arguments to pass to the
    subcommand decorator.
    """
    return (list(name_or_flags), kwargs)


# Start subcommand logic.
@subcommand([argument("--db", type=str, help="Path to database directory", default="~/.cache/img_embed"), argument("dir", type=str, help="Directory to embed images from")])
def embed_dir(args):
    from database_connection import EmbedImgDir
    import os.path as path
    from transformers import (
        AutoProcessor,
        AutoModel
    )
    import torch
    import os
    if not path.exists("~/.cache/img_embed") and args.db == "~/.cache/img_embed/embeds.db":
        os.makedirs(os.path.expanduser("~/.cache/img_embed"))
        os.makedirs(os.path.expanduser("~/.cache/img_embed/") + path.basename(args.dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "google/siglip-so400m-patch14-384"
    model = AutoModel.from_pretrained(model_name, device_map=device)
    processor = AutoProcessor.from_pretrained(model_name, device_map=device)
    embedder = EmbedImgDir(args.dir, model_name, processor, model, args)
    embedder.process_dir()


@subcommand([argument("--db", help="Directory to find the database at."), argument("--num-results", type=int, help="How many results to bring back."), argument("text", type=str, help="Text query to match up to images by similarity")])
def query(args):
    from transformers import (
        AutoTokenizer,
        AutoModel,
        AutoProcessor
    )
    from database_connection import EmbedImgDir
    import torch
    from PIL import Image

    # setup the models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "google/siglip-so400m-patch14-384"
    model = AutoModel.from_pretrained(model_name, device_map=device)
    processor = AutoProcessor.from_pretrained(model_name, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=device)

    embedder = EmbedImgDir(args.db, model_name, processor, model, args)

    num_results = args.num_results if args.num_results else 5

    results = embedder.query_text(args.text, tokenizer, args.db, num_results)

    for i, result in enumerate(results["metadatas"][0]):
        print(f"Result {i}: {result['path']}")
    
    print("Displaying first result...")
    Image.open(results["metadatas"][0][0]["path"]).show()

@subcommand([argument("--db", help="Directory to find the database at."), argument("--num-results", type=int, help="How many results to bring back."), argument("image", type=str, help="Path to image to use as query")])
def query_image(args):
    from transformers import (
        AutoModel,
        AutoProcessor
    )
    from database_connection import EmbedImgDir
    import torch
    from PIL import Image

    # setup the models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "google/siglip-so400m-patch14-384"
    model = AutoModel.from_pretrained(model_name, device_map=device)
    processor = AutoProcessor.from_pretrained(model_name, device_map=device)

    embedder = EmbedImgDir(args.db, model_name, processor, model, args)

    num_results = args.num_results if args.num_results else 5

    img = Image.open(args.image)
    results = embedder.query_image(img, num_results)
    
    for i, result in enumerate(results["metadatas"][0]):
        print(f"Result {i}: {result['path']}")
    
    print("Displaying first result...")
    Image.open(results["metadatas"][0][0]["path"]).show()

if __name__ == "__main__":
    args = cli.parse_args()
    if args.subcommand is None:
        cli.print_help()
    else:
        args.func(args)