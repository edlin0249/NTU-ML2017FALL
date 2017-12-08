#!/bin/bash
wget 'https://www.dropbox.com/s/4s6orizw09qulyx/word2vec.model?dl=1'
wget 'https://www.dropbox.com/s/21i54xxitidyqob/model-W2V-LSTM30.h5?dl=1'
mv word2vec.model?dl=1 word2vec.model
mv model-W2V-LSTM30.h5?dl=1 model-W2V-LSTM30.h5
python3 hw4_test.py $1 $2