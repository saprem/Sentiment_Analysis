# Sentiment_Analysis-and-Emojify

## LSTM-Sentiment-Analysis

1. Sentiment_LSTM.ipynb  
IPython notebook with training and testing code with text-blogs explaining the approach.


2. Sentiment_LSTM_Code.py
Python code(same as above)

3. Datasets
negativeReviews and PositiveReviews 

4. idsMatrix.npy , wordList.npy , wordVectors.npy
Numpy array loaded used in code above. They can be calculated while training on first time and then saved so to save time while re-training and directly loaded.

5.  models_trained
Saved models 


## Emojify

1. Emojify.ipynb and Emojify.py
Code for training testing with text-blogs explaining the approach.

2. data 
Training and testing data as well the pre-trained GloVe embeddings useful while training.

3. emo_utils.py and emo_utils.pyc
Has helper functions such as label_to_emoji(),read_glove_vecs().


## SentimentAnalysis_compare.ipynb
Compares accuracy using various algortihms on IMDB dataset. 

