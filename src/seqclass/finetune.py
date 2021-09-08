# import libraries
import pip
import time

# import packages if not exist
def import_or_install(package):
    try:
        __import__(package)
    except ImportError:
        pip.main(['install', package])  

import_or_install('transformers')
import torch
from sklearn.metrics import accuracy_score
from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, AdamW 

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

    def __init__(self, num_labels, tokenizer='allenai/longformer-base-4096', model='allenai/longformer-base-4096'):
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer = LongformerTokenizerFast.from_pretrained(tokenizer)
        self.model = LongformerForSequenceClassification.from_pretrained(model, num_labels=num_labels, output_hidden_states=False).to(self.device)
        for param in self.model.longformer.encoder.parameters(): 
            param.requires_grad = False

#-----------------------------------------------------------------

    def train(self, train_loader, val_loader, save_name, learning_rate=5e-5, max_epoch=3):

        optim = AdamW(self.model.parameters(), lr=learning_rate)
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
            modelName = "./models/" + save_name + "-" + str(round(time.time()))
            self.model.save_pretrained(modelName)
            #print("Model " + str(round(time.time())) + " saved!")
            self._print_results("Train", train_loss, train_acc, train_loader)

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
            self._print_results("Val", val_loss, val_acc, val_loader)
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

        self._print_results("Test", test_loss, test_acc, test_loader)
        
        return true_labels, pred_labels

#-----------------------------------------------------------------

    def _print_time(self, tag, start, end):
        print(tag, round((end-start)//3600), "hr", round(((end-start)%3600)//60), "min",  round((end-start)%60), "sec")     

#-----------------------------------------------------------------

    def _print_results(self, type, loss, acc, data_loader):
        print(
            type + " loss:", round(loss/(len(data_loader)*data_loader.batch_size), 4), "\t", 
            type + " acc:", round(acc/(len(data_loader)*data_loader.batch_size), 4), 
        )

#-----------------------------------------------------------------
#  End of Class LongformerClassifier
#-----------------------------------------------------------------

if __name__ == "__main__":

    import pandas as pd
    from config import *
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    from torch.utils.data import DataLoader

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--checkpoint', help="specify checkpoint using", default="allenai/longformer-base-4096")
    parser.add_argument('--save_name', help="name for saving models", required=True)
    args, unknown = parser.parse_known_args()

    # set parameters
    N_LABELS = 19
    MAX_LENGTH = 3072
    LR = 2e-5              
    BATCH_SIZE = 2
    MAX_EPOCH = 3

    # define classifier
    clf = LongformerClassifier(num_labels=N_LABELS, tokenizer='allenai/longformer-base-4096', model='models/'+args.checkpoint)
    
    # import datasets
    train_set = pd.read_json(TRAIN_DIR)
    val_set = pd.read_json(VAL_DIR)
    test_set = pd.read_json(TEST_DIR)

    # encode texts with tokenizer
    print("Encoding training set ...")
    train_encodings = clf.tokenizer(list(train_set.X), max_length=MAX_LENGTH, truncation=True, padding=True)
    print("Encoding validation set ...")
    val_encodings = clf.tokenizer(list(val_set.X), max_length=MAX_LENGTH, truncation=True, padding=True)
    print("Encoding test set ...")
    test_encodings = clf.tokenizer(list(test_set.X), max_length=MAX_LENGTH, truncation=True, padding=True)

    # create numerical index 
    class2label = {cls:i for i, cls in enumerate(sorted(list(set(train_set.y_class))))}
    label2class = {v:k for k,v in class2label.items()}

    # build custom datasets
    train_dataset = GutenbergDataset(train_encodings, [class2label[cls] for cls in train_set.y_class])
    val_dataset = GutenbergDataset(val_encodings, [class2label[cls] for cls in val_set.y_class])
    test_dataset = GutenbergDataset(test_encodings, [class2label[cls] for cls in test_set.y_class])

    # build data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # train model
    print("========== Start Training ==========")
    clf.train(train_loader, val_loader, save_name=args.save_name, learning_rate=LR, max_epoch=MAX_EPOCH)
    print("========== Start Testing ===========")
    _ , _ = clf.predict(test_loader)