# Sentence-Ordering


## Steps to follow:
* Install latest [Anaconda](https://docs.anaconda.com/anaconda/install/)
* Create provided environment by running : conda env create --file environment.yml --name <desired-name>
* Create Tokenizer, Pretrained, Models and relevant data folder
  * Create required subfolders for processed results of data(Refer to data preparation notebooks) eg: processed, original and permutations for arxiv_data 
  * Download [Glove Embeddings](http://nlp.stanford.edu/data/wordvecs/glove.6B.zip) and place unzipped folder glove.6B in Pretrained folder. (Only needed if running pretrained word embedding experiment).
* Place data in relevant data folder and run corresponding preparation code from jupyter notebooks(Currently Earthquake articles, Arxiv Abstracts and NIPs abstracts).
  Create required subfolder(eg: processed, )
* Run Train.py file using : python Train.py
  * Contains a number of flags that can be set differently to run different experiments(eg : useWordEmbeddings, usePretrainedWordEmbeddings, split_at etc.)
