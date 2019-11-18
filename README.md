# Sentence-Ordering


## Steps to follow:
* Install latest [Anaconda](https://docs.anaconda.com/anaconda/install/)
* Create provided environment by running : conda env create --file environment.yml --name "desired-name"
* Create Pretrained and relevant data folder. 
  * Download [Glove Embeddings](http://nlp.stanford.edu/data/wordvecs/glove.6B.zip) and place unzipped folder glove.6B in Pretrained folder. (Only needed if running pretrained word embedding experiment).
* Place data in relevant data folder and run corresponding preparation code from jupyter notebooks(Currently Earthquake articles, Arxiv Abstracts and NIPs abstracts).
* Run Train.py file using : python Train.py
  * Contains a number of flags that can be set differently to run different experiments(eg : useWordEmbeddings, usePretrainedWordEmbeddings, split_at etc.)
  * Required : Fill out location of processed data files and name(.h5 file) of final trained model.

## TODO
* PointerLSTM code is still a work in progress and doesn't currently work as expected.
