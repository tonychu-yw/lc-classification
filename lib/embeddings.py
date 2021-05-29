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

    def __init__(self, model="longformer"):

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if model == "longformer":
            self.tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096', output_hidden_states=True)
            self.model = LongformerModel.from_pretrained('allenai/longformer-base-4096').to(self.device)

#-----------------------------------------------------------------

    def _embed(self, book):

        input_ids = self.tokenizer(book, padding=True, truncation=True, return_tensors="pt").input_ids.to(self.device)
        outputs = self.model(input_ids)

        # extract last hidden state
        # print(outputs.keys())
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
        # print(last_hidden_states.shape)

        # get document embeddings
        #pooling = torch.nn.MaxPool1d(768, stride=1)
        #doc_embedding = pooling(last_hidden_states).squeeze().cpu().detach().numpy()
        doc_embedding = list(torch.mean(last_hidden_states[0], dim=0).squeeze().cpu().detach().numpy())
        #print(doc_embedding.shape)

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
            if iter%500 == 0:
                end = time.time()
                print(iter, "- time:", round((end-start)//60), "min",  round((end-start)%60), "sec")

        end = time.time()
        print("Total runtime:", round((end-start)//60), "min",  round((end-start)%60), "sec")

        return doc_embeds

#-----------------------------------------------------------------

    @classmethod
    def save_embeddings(embeddings, filename):
        with open(filename, 'wb') as handle:
            pickle.dump(np.array(embeddings), handle)

#-----------------------------------------------------------------
#  End of Class Book2Vec
#-----------------------------------------------------------------
