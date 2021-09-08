import numpy as np
import time
import_or_install('transformers')
import torch
from transformers import LongformerTokenizer, LongformerModel
from config import *

#-----------------------------------------------------------------
#  Class Book2Vec
#-----------------------------------------------------------------

class Book2Vec:
  
    def __init__(self, tokenizer='allenai/longformer-base-4096', model='allenai/longformer-base-4096'):

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer = LongformerTokenizer.from_pretrained(tokenizer)
        self.model = LongformerModel.from_pretrained(model, output_hidden_states=True).to(self.device)

#-----------------------------------------------------------------

    def _embed(self, book):

        input_ids = self.tokenizer(book, padding=True, truncation=True, return_tensors="pt").input_ids.to(self.device)
        outputs = self.model(input_ids)

        # extract last hidden state
        #last_hidden_states = outputs[0]
        #doc_embedding = list(torch.mean(last_hidden_states[0], dim=0).squeeze().cpu().detach().numpy())

        # extract last 4 hidden states
        last_four_layers = [outputs.hidden_states[i] for i in (-1, -2, -3, -4)]
        cat_hidden_states = torch.cat(tuple(last_four_layers), dim=-1)
        doc_embedding = list(torch.mean(cat_hidden_states, dim=1).squeeze().cpu().detach().numpy())

        return doc_embedding

#-----------------------------------------------------------------

    def get_embeddings(self, book_list):

        doc_embeds = []
        start = time.time()
        iter = 0
        for book_i in book_list:
            doc_embedding = self._embed(book_i)
            doc_embeds.append(doc_embedding)
            iter += 1
            if iter%1000 == 0:
                end = time.time()
                print(iter, "- time:", round((end-start)//60), "min",  round((end-start)%60), "sec")
        end = time.time()
        print("Total runtime:", round((end-start)//60), "min",  round((end-start)%60), "sec")

        return doc_embeds

#-----------------------------------------------------------------
#  End of Class Book2Vec
#-----------------------------------------------------------------

if __name__ == "main":

    # import data
    model = Book2Vec(tokenizer='allenai/longformer-base-4096', model='allenai/longformer-base-4096')
    train_set = pd.read_json(TRAIN_DIR)
    val_set = pd.read_json(VAL_DIR)
    test_set = pd.read_json(TEST_DIR)

    # get embeddings
    train_embeddings = model.get_embeddings(train_set.X)
    val_embeddings = model.get_embeddings(val_set.X)
    test_embeddings = model.get_embeddings(test_set.X)

    # output embeddings to dataframe
    train_embeddings_df = pd.DataFrame({"id": train_set.id, "embeddings": train_embeddings})
    val_embeddings_df = pd.DataFrame({"id": val_set.id, "embeddings": val_embeddings)})
    test_embeddings_df = pd.DataFrame({"id": test_set.id, "embeddings": test_embeddings})

    # save embeddings (dimension = 768)
    train_embeddings_df.to_json('./work/longformer_train_embeddings.json')
    val_embeddings_df.to_json('./work/longformer_val_embeddings.json')
    test_embeddings_df.to_json('./work/longformer_test_embeddings.json')
    print("Longformer embeddings saved to ./work" )
