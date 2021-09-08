import pandas as pd
import_or_install('sentence_transformers')
from sentence_transformers import SentenceTransformer
from config import *

if __name__ = "main":

    model = SentenceTransformer('all-MiniLM-L6-v2')
    model.max_seq_length = 512

    # read text data
    train_set = pd.read_json(TRAIN_DIR)
    val_set = pd.read_json(VAL_DIR)
    test_set = pd.read_json(TEST_DIR)

    # encode text to embeddings
    train_embeddings = model.encode(train_set.X)
    val_embeddings = model.encode(val_set.X)
    test_embeddings = model.encode(test_set.X)

    # output embeddings to dataframe
    train_embeddings_df = pd.DataFrame({"id": train_set.id, "embeddings": list(train_embeddings)})
    val_embeddings_df = pd.DataFrame({"id": val_set.id, "embeddings": list(val_embeddings)})
    test_embeddings_df = pd.DataFrame({"id": test_set.id, "embeddings": list(test_embeddings)})

    # save embeddings (dimension = 384)
    train_embeddings_df.to_json('./work/sbert_train_embeddings.json')
    val_embeddings_df.to_json('./work/sbert_val_embeddings.json')
    test_embeddings_df.to_json('./work/sbert_test_embeddings.json')
    print("SBERT embeddings saved to ./work" )