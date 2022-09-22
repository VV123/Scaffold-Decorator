import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim import SGD, Adam
from torch.nn import MSELoss, L1Loss
from torch.nn.init import xavier_uniform_
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from transformers import BertTokenizer, BertModel
import numpy as np
import sys
from model_auto import Seq2SeqTransformer, PositionalEncoding, generate_square_subsequent_mask, create_mask
from utils import top_k_top_p_filtering, open_file, read_csv_file, load_sets
import vocabulary as mv
import dataset as md
import torch.utils.data as tud
import os.path
import glob
import math
import torch
import torch.nn as nn
from collections import Counter
from torch import Tensor
import io
import time

torch.manual_seed(0)

def evaluate(model, val_iter):
    model.eval()
    losses = 0
    for idx, (src, tgt) in (enumerate(valid_iter)):
        src = src[0].transpose(0, 1).to(device)
        tgt = tgt[0].transpose(0, 1).to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,
                              src_padding_mask, tgt_padding_mask, src_padding_mask)
        tgt_out = tgt[1:,:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
    return losses / len(val_iter)


def train_epoch(model, train_iter, optimizer):
    model.train()
    losses = 0
    for idx, (src, tgt) in enumerate(train_iter):
        src = src[0].transpose(0, 1).to(device)
        tgt = tgt[0].transpose(0, 1).to(device)
            
        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,
                                src_padding_mask, tgt_padding_mask, src_padding_mask)
      
        optimizer.zero_grad()
      
        tgt_out = tgt[1:,:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        loss.backward()

        optimizer.step()
        if idx % 500 == 0:
            print('Train Epoch: {}\t Loss: {:.6f}'.format(epoch, loss.item()))     
        losses += loss.item()

    print('====> Epoch: {0} total loss: {1:.4f}.'.format(epoch, losses))
    return losses / len(train_iter)


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1):
        memory = memory.to(device)
        memory_mask = torch.zeros(ys.shape[0], memory.shape[0]).to(device).type(torch.bool)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                                    .type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        
        pred_proba_t = topk_filter(prob, top_k=30) #[b, vocab_size]
        probs = pred_proba_t.softmax(dim=1) #[b, vocab_size]
        next_word = torch.multinomial(probs, 1)
        #_, next_word = torch.max(prob, dim = 1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
          break
    return ys

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--mode', choices=['train', 'infer', 'baseline'],\
        default='train',help='Run mode')
    arg_parser.add_argument('--device', choices=['cuda', 'cpu'],\
        default='cuda',help='Device')
    arg_parser.add_argument('--epoch', default='100', type=int)
    arg_parser.add_argument('--batch_size', default='512', type=int)
    arg_parser.add_argument('--layer', default=3, type=int)
    arg_parser.add_argument('--path', default='model_chem.h5', type=str)
    arg_parser.add_argument('--datamode', default=1, type=int)
    arg_parser.add_argument('--d_model', default=512, type=int)
    arg_parser.add_argument('--nhead', default=8, type=int)
    arg_parser.add_argument('--embedding_size', default=200, type=int)
    arg_parser.add_argument('--loadmodel', default=False, action="store_true")
    arg_parser.add_argument("--loaddata", default=False, action="store_true")
    args = arg_parser.parse_args()

    print('========== [Prediction] Transformer ==============')

    scaffold_list, decoration_list = zip(*read_csv_file('zinc/chembl_train.smi', num_fields=2))
    vocabulary = mv.DecoratorVocabulary.from_lists(scaffold_list, decoration_list)
    training_sets = load_sets('zinc/chembl_train.smi')
    dataset = md.DecoratorDataset(training_sets, vocabulary=vocabulary)

    BATCH_SIZE = args.batch_size
    SRC_VOCAB_SIZE = dataset.vocabulary.len()[0]
    TGT_VOCAB_SIZE = dataset.vocabulary.len()[1]

    EMB_SIZE = args.d_model
    NHEAD = args.nhead
    FFN_HID_DIM = 512

    NUM_ENCODER_LAYERS = args.layer
    NUM_DECODER_LAYERS = args.layer
    NUM_EPOCHS = args.epoch
    PAD_IDX = 0
    BOS_IDX = 2
    EOS_IDX = 1
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = args.device

    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, 
                                 EMB_SIZE, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
                                 FFN_HID_DIM, args=args)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    
    num_train= int(len(dataset)*0.8)
    num_test= len(dataset) -num_train
    train_data, test_data = torch.utils.data.random_split(dataset, [num_train, num_test])
    train_iter = tud.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=md.DecoratorDataset.collate_fn, drop_last=True)
    test_iter = tud.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=md.DecoratorDataset.collate_fn, drop_last=True)
    valid_iter = test_iter

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    optimizer = torch.optim.Adam(
        transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95) 
    if args.mode == 'train':
        transformer = transformer.to(DEVICE)

        if args.loadmodel:
            transformer.load_state_dict(torch.load(args.path))


        min_loss, val_loss = 100000000, 100000000
        for epoch in range(1, NUM_EPOCHS+1):
            start_time = time.time()
            train_loss = train_epoch(transformer, train_iter, optimizer)
            scheduler.step()
            end_time = time.time()
            if (epoch+1)%1==0:
                val_loss = evaluate(transformer, valid_iter)
                if val_loss < min_loss:
                    min_loss = val_loss
                    torch.save(transformer.state_dict(), args.path)
                    print('Model saved!') 

            print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "
                f"Epoch time = {(end_time - start_time):.3f}s"))
    
 
    elif args.mode == 'infer':
        if args.device == 'cpu':
            transformer.load_state_dict(torch.load(args.path,  map_location=torch.device('cpu')))
        else:
            transformer.load_state_dict(torch.load(args.path))
        device = args.device
        transformer.to(device)
       

        transformer.eval()
        val_data, _ = torch.utils.data.random_split(test_data, [2, num_test-2])
        val_dataloader = tud.DataLoader(val_data, batch_size=1, shuffle=True, collate_fn=md.DecoratorDataset.collate_fn, drop_last=False)

        for i, (x_dat, y_dat) in enumerate(val_dataloader):
            x1 = x_dat[0].to(device)
            y = y_dat[0].to(device)
            x1 = x1.transpose(0, 1)
            y = y.transpose(0, 1)
            src = x1 #= (torch.LongTensor(tokens).reshape(num_tokens, 1) )
            src_mask = (torch.zeros(x1.shape[0], x1.shape[0])).type(torch.bool)
            ybar = greedy_decode(transformer, src, src_mask, max_len= 50, start_symbol=BOS_IDX).flatten()
            print(ybar)
            ybar = vocabulary.decode_decoration(ybar.to('cpu').data.numpy())
            print('y prediction')
            print(ybar)
            print('y gold')
            print(y.squeeze().to('cpu').data.numpy())
            yg_s =  vocabulary.decode_decoration(y.squeeze().to('cpu').data.numpy())
            print(yg_s)
       
