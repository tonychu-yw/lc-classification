# import libraries
import time
import pickle
import numpy as np
import pandas as pd
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import subprocess
import sys
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install('transformers')
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, AdamW 

#-----------------------------------------------------------------

# save pickle files
def save_pickle(stuff, fileName):
    with open(fileName, 'wb') as f:
        pickle.dump(stuff, f, pickle.HIGHEST_PROTOCOL)

#-----------------------------------------------------------------

# load pickle files
def load_pickle(fileName):
    with open(fileName, 'rb') as f:
        return pickle.load(f)

#-----------------------------------------------------------------

#-----------------------------------------------------------------
#  Class GutenbergDataset
#-----------------------------------------------------------------

class GutenbergDataset(torch.utils.data.Dataset):
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.as_tensor(self.labels[idx], dtype=torch.int64)
        return item

    def __len__(self):
        return len(self.labels)

#-----------------------------------------------------------------
#  End of Class GutenbergDataset
#-----------------------------------------------------------------

#-----------------------------------------------------------------
#  Class LongformerClassifier
#-----------------------------------------------------------------

class LongformerClassifier:

    def __init__(self, tokenizer='allenai/longformer-base-4096', model='allenai/longformer-base-4096', num_labels=19):
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer = LongformerTokenizerFast.from_pretrained(tokenizer)
        self.model = LongformerForSequenceClassification.from_pretrained(model, num_labels=num_labels, output_hidden_states=False).to(self.device)
        for param in self.model.longformer.encoder.parameters(): 
            param.requires_grad = False

#-----------------------------------------------------------------

    def train(self, train_loader, val_loader, save_model_name, max_epoch=3):

        optim = AdamW(self.model.parameters(), lr=5e-5)
        start = time.time()
        mid_prev = start
        
        for epoch in range(max_epoch):
            
            print("epoch " + str(epoch))
            train_loss = 0
            val_loss = 0
            train_acc = 0
            val_acc = 0

            # train set
            self.model.train()
            for batch in train_loader:
                optim.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]
                loss.backward()
                optim.step()
                
                # record train loss and acc
                train_loss += loss.data.item()
                true = labels.tolist()
                pred = outputs.logits.argmax(-1).tolist()
                train_acc += accuracy_score(true, pred, normalize=False)

            # save model
            modelName = "./models/" + save_model_name + "-" + str(round(time.time()))
            self.model.save_pretrained(modelName)
            #print("Model " + str(round(time.time())) + " saved!")
            
            print(
                "Train loss:", round(train_loss/(len(train_loader)*train_loader.batch_size), 4), "\t", 
                "Train acc:", round(train_acc/(len(train_loader)*train_loader.batch_size), 4)
            )

            # validation set
            self.model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs[0]

                    # record val loss and acc
                    val_loss += loss.data.item()
                    true = labels.tolist()
                    pred = outputs.logits.argmax(-1).tolist()
                    val_acc += accuracy_score(true, pred, normalize=False)
            
            # record epoch runtime 
            mid_curr = time.time()
            print(
                "Val loss:", round(val_loss/(len(val_loader)*val_loader.batch_size), 4), "\t",
                "Val acc:", round(val_acc/(len(val_loader)*val_loader.batch_size), 4)
            )
            self._print_time("Runtime:", mid_prev, mid_curr)
            mid_prev = mid_curr
            print("--------------------")

        end = time.time()
        self._print_time("Total Runtime", start, end)

#-----------------------------------------------------------------

    def predict(self, test_loader):
        
        true_labels = []
        pred_labels = []
        test_loss = 0
        test_acc = 0

        self.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]
                
                # record test loss and acc
                test_loss += loss.data.item()
                true = labels.tolist()
                pred = outputs.logits.argmax(-1).tolist()
                test_acc += accuracy_score(true, pred, normalize=False)
                true_labels.extend(true)
                pred_labels.extend(pred)

        print(
            "Test loss:", round(test_loss/(len(test_loader)*test_loader.batch_size), 4), "\t", 
            "Test acc:", round(test_acc/(len(test_loader)*test_loader.batch_size), 4), 
        )
        
        return true_labels, pred_labels

#-----------------------------------------------------------------

    def _print_time(self, tag, start, end):
        print(tag, round((end-start)//3600), "hr", round(((end-start)%3600)//60), "min",  round((end-start)%60), "sec")     

#-----------------------------------------------------------------
#  End of Class LongformerClassifier
#-----------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', help="train or test model", required=True)
    parser.add_argument('--checkpoint', help="specify checkpoint using", default="allenai/longformer-base-4096")
    args, unknown = parser.parse_known_args()

    # set parameters
    BATCH_SIZE = 2

    # import encodings (X)
    train_encodings = load_pickle('work/train_encodings.pkl')
    val_encodings = load_pickle('work/val_encodings.pkl')
    test_encodings = load_pickle('work/test_encodings.pkl')

    # import labels (y)
    train_labels = load_pickle('work/train_class.pkl')
    val_labels = load_pickle('work/val_class.pkl')
    test_labels = load_pickle('work/test_class.pkl')

    # create numerical index 
    class2label = {cls:i for i, cls in enumerate(sorted(list(set(train_labels))))}
    label2class = {v:k for k,v in class2label.items()}

    # build custom datasets
    train_dataset = GutenbergDataset(train_encodings, [class2label[cls] for cls in train_labels])
    val_dataset = GutenbergDataset(val_encodings, [class2label[cls] for cls in val_labels])
    test_dataset = GutenbergDataset(test_encodings, [class2label[cls] for cls in test_labels])

    # build data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # execute main
    clf = LongformerClassifier(tokenizer='allenai/longformer-base-4096', model=args.checkpoint, num_labels=19)
    if args.mode == "train":
        print("========== Start Training ==========")
        clf.train(train_loader, val_loader, save_model_name="longformer-class-2048", max_epoch=3)
        print("====================================")
    elif args.mode == "test":
        print("========== Start Testing ==========")
        true, pred = clf.predict(test_loader)
        print("===================================")
    else:
        print('Error: Please input "train" or "test".')


