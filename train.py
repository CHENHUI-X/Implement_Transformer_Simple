import argparse
from model import Transformer
import os
import string
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ISCUDA = torch.cuda.is_available()



def parameter_parse():
    FLAGS = argparse.ArgumentParser()
    FLAGS.add_argument('--dataroot' ,
                       help='Where is the data')

    FLAGS.add_argument('--embedding-dim', type=int,default = 512,
                       help="Embedded dimension")

    FLAGS.add_argument('--head-num', type=int,default = 8,
                       help="Embedded dimension")

    FLAGS.add_argument('--num-layers', type=int,default = 1,
                       help="Number of block layers")

    FLAGS.add_argument('--epochs', type=int,default = 100,
                       help="Epoch when training ")

    FLAGS.add_argument('--batch-size', type=int,default = 8,
                       help="Batch size when training ")

    FLAGS.add_argument('--val-size', type = float, default= 0.2,
                       help="Size for val ")

    FLAGS.add_argument('--lr', type=float, default=0.001,
                       help="Learning rate when training")

    FLAGS.add_argument('--save', type=str, default='logs',
                       help="Location to logs and ckpts")

    # action = 'store_true'的时候,表示只要加上 --gpu 则默认为true
    FLAGS.add_argument('--gpu', action='store_true',
                       help="Set this to use GPU, default use CPU")

    FLAGS.add_argument('--n-workers', type=int, default=1,
                       help = "How many processes for preprocessing")

    FLAGS.add_argument('--random-state', type=int, default=0,
                       help="Random_state")

    return FLAGS

def get_sentence(opt : any ) -> tuple[list,list] :

    data_root = opt.dataroot
    # 3 files should be fetched from data_root,
    # one holds the positive text and one is the negative
    # and finally a license.txt

    neg_file_path = data_root + '\\neg\\'
    neg_file_list = os.listdir(neg_file_path)

    pos_file_path = data_root + '\\pos\\'
    pos_file_list = os.listdir(pos_file_path)


    # 1 . the sentence save as binary text
    # we need to convert it to string
    # 2 . we need remove '\n' and punctuation
    # 3 . we split the  str according whitespace

    neg_sentence = []
    # [[str],[str],[str],[str]]
    for file in neg_file_list:
        with open(neg_file_path + file, 'rb') as f:
            neg_sentence += [
                i.decode('ascii') \
                    .strip() \
                    .translate(
                    str.maketrans('', '', string.punctuation)
                ).split()
                for i in f.readlines()]

    pos_sentence = []
    for file in pos_file_list:
        with open(pos_file_path + file, 'rb') as f:
            pos_sentence += [
                i.decode('ascii') \
                    .strip() \
                    .translate(
                    str.maketrans('', '', string.punctuation)
                ).split()
                for i in f.readlines()]

    return neg_sentence , pos_sentence


def text2numeric(neg_sentence_list,pos_sentence_list) :

    # step 1 : collect all texts
    sentence_list = neg_sentence_list + pos_sentence_list
    all_words = []
    # get all words [word1,word2,word3...]
    for i in sentence_list : all_words += i

    # step 2 : deduplication and sort
    das_words = ['EOS'] + sorted(list(set(all_words)))

    # step 3 : calculate the total amount of all vocabulary
    #    and the maximum sentence length
    vocabulary_size = len(das_words) + 1
    max_seq_len = max([len(i) + 1 for i in sentence_list])
    print(max_seq_len)

    # step 4 : number the word
    encoder = LabelEncoder()
    encoder.fit(das_words)

    # step 5 : word 2 index  and calculate the length for sentence
    neg_sentence_list_with_index = \
        [list(encoder.transform(i)+1) for i in neg_sentence_list]
    neg_sentence_len = [len(i) for i in neg_sentence_list]

    # step 6 : fill the sentence with 0
    # [1,2,3,4,0,0,0,...]
    neg_sentence_list_with_index_and_fill = \
        [ index_list + [0] * (max_seq_len - le)
         for index_list ,le in zip(
            neg_sentence_list_with_index, neg_sentence_len)]

    # same as above
    pos_sentence_list_with_index = \
        [list(encoder.transform(i)+1) for i in pos_sentence_list]
    pos_sentence_len = [len(i) for i in pos_sentence_list]
    pos_sentence_list_with_index_and_fill = \
        [index_list + [0] * (max_seq_len - le) for index_list, le in
         zip(pos_sentence_list_with_index, pos_sentence_len)]

    # step 7 : prepare label
    # [
    #     [0 , 0 , 0 , 0 , 0 , 0 , 0...] ,
    #     [0 , 0 , 0 , 0 , 0 , 0 , 0...] ,
    #     [0 , 0 , 0 , 0 , 0 , 0 , 0...] ,
    # ]
    neg_label = [[0]*max_seq_len] * len(neg_sentence_list_with_index_and_fill)
    pos_label = [[1]*max_seq_len] * len(pos_sentence_list_with_index_and_fill)



    return neg_sentence_list_with_index_and_fill,neg_label,\
           pos_sentence_list_with_index_and_fill,pos_label , \
           vocabulary_size ,max_seq_len



class Textdataset(Dataset):
    def __init__(self,
                 sentence_with_index,
                 label,
                 ):
        self.X = sentence_with_index
        self.Y = label

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]) , torch.tensor(self.Y[idx])

    def __len__(self):
        return len(self.Y)


def get_loader(opt,
               neg_sentence_list_with_index, neg_label,
               pos_sentence_list_with_index, pos_label ):
    sentence_list_with_index = \
        neg_sentence_list_with_index + pos_sentence_list_with_index
    label = neg_label + pos_label

    X_train, X_val, y_train, y_val = train_test_split(
        sentence_list_with_index, label,
        test_size = opt.val_size, random_state = opt.random_state
    )

    training_data = Textdataset(
        X_train,y_train
    )

    val_data = Textdataset(
        X_val,y_val
    )

    train_dataloader = DataLoader(
        training_data, batch_size=opt.batch_size,
        shuffle=True,num_workers=opt.n_workers)

    val_dataloader = DataLoader(
        val_data, batch_size = opt.batch_size,
        shuffle=True,num_workers = opt.n_workers)

    return train_dataloader,val_dataloader

def train_and_val(opt,
          train_dataloader,val_dataloader,
          vocabulary_size  , max_seq_len):

    vocabulary_size = vocabulary_size
    target_vocab_size = 2 # for this work , target just positive or negative
    seq_length = max_seq_len

    print('Model initialization ....\n')
    model = Transformer(
        embed_dim= opt.embedding_dim,
        src_vocab_size= vocabulary_size,
        target_vocab_size= target_vocab_size,
        seq_length= seq_length,
        num_layers= opt.num_layers,
        expansion_factor = 4, n_heads= opt.head_num)

    if ISCUDA and opt.gpu:
        model = model.cuda()

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr = opt.lr,
        weight_decay=0.001, momentum=0.9
    )

    train_losses = []
    train_acc = []
    val_losses = []
    step = 0
    for epoch in range(opt.epochs):

        loss_epoch_train_for_batch = []
        loss_epoch_val_for_batch = []

        acc_epoch_train_for_batch = []
        # Run the training batches
        for batch_index, (X_train, y_train) in tqdm(
              enumerate(train_dataloader), total=len(train_dataloader)
        ):
            step += 1
            # out shape with (batch , seq_len , word_size)
            out = model(X_train,y_train)
            # get pro of latest token in latest sequence
            y_pro = out.type(torch.FloatTensor)[:,-1,-1]
            # get label
            pre_label = [ 1 if i >= 0.5 else 0 for i in y_pro ]
            # get label of latest token in every sequence
            y = y_train.type(torch.FloatTensor)[:,-1] # y_train shape is (batch_size,  seq_len)
            # that is we just predict probabilities of latest token of each sentence
            # as label of this sentence
            print(pre_label,y)
            acc = accuracy_score(y, pre_label)
            loss = criterion(y_pro.unsqueeze(1), y.unsqueeze(1))

            loss_epoch_train_for_batch.append(loss.item())
            acc_epoch_train_for_batch.append(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('\n Epoch [{}/{}] - step {} - Loss: {:.4f} - Acc: {:.4f} '
                  .format(epoch + 1, opt.epochs, step, loss.item() , acc))

        train_losses.append(np.mean(loss_epoch_train_for_batch))
        train_acc.append(np.mean(acc_epoch_train_for_batch))
        print(
            f'\n Epoch [{epoch+1:2}/{opt.epochs:2}] ' + \
            f'-- Average_loss: { np.mean(loss_epoch_train_for_batch):6.4f} ' + \
            f'-- Average_acc: { np.mean(acc_epoch_train_for_batch):6.4f} '
        )

        # # Run the validate  batches
        with torch.no_grad():
            for X_val, y_val in val_dataloader:
                out_val = model(X_val,y_val)

                y_val_pro = out_val.type(torch.FloatTensor)[:, -1, -1]
                val_pre_label = [1 if i >= 0.5 else 0 for i in y_val_pro]
                val_y_label = y_val.type(torch.FloatTensor)[:, -1]
                val_acc = accuracy_score(val_y_label,val_pre_label)
                val_loss = criterion(y_val_pro.unsqueeze(1), val_y_label.unsqueeze(1))

                loss_epoch_train_for_batch.append(val_loss.item())
                acc_epoch_train_for_batch.append(val_acc)

                loss_epoch_val_for_batch.append(val_loss.item())

        val_losses.append(np.mean(loss_epoch_val_for_batch))
        print(f'\nEpoch: {epoch} Val Loss: {np.mean(loss_epoch_val_for_batch):10.8f} ')


def main():
    FLAGS = parameter_parse()
    args, unparsed = FLAGS.parse_known_args()

    assert not unparsed , "Argument {} not recognized".format(unparsed)
    assert not (args.embedding_dim % args.head_num),\
        'The number of heads should be divisible by the embedding dimension'

    # get sentence and transform to index in vocabulary
    print('-----------------get sentence ... --------------------\n ')
    neg_sentence, pos_sentence = get_sentence(args)

    print('-----------------sentence to index ... --------------------\n ')
    neg_sentence_list_with_index, neg_label, \
    pos_sentence_list_with_index, pos_label, \
    vocabulary_size  ,max_seq_len = \
        text2numeric(neg_sentence, pos_sentence)
    # print(vocabulary_size)
    # get dataloader
    train_dataloader,val_dataloader = \
        get_loader(args,
                   neg_sentence_list_with_index, neg_label,
                   pos_sentence_list_with_index, pos_label)

    # then train it
    print('-----------------start train  ... --------------------\n ')
    train_and_val(args,train_dataloader,val_dataloader,vocabulary_size,max_seq_len)



if __name__ == '__main__':
    main()