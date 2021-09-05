import numpy as np
import time
import pickle

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install('transformers')
import torch
from transformers import LongformerTokenizer, LongformerModel

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

    @classmethod
    def save_embeddings(cls, embeddings, filename):
        with open(filename, 'wb') as handle:
            pickle.dump(np.array(embeddings), handle)
            
#-----------------------------------------------------------------

    @classmethod
    def load_embeddings(cls, filename):
        with open(filename, 'rb') as handle:
            embeddings = pickle.load(handle)
        return embeddings

#-----------------------------------------------------------------
#  End of Class Book2Vec
#-----------------------------------------------------------------
