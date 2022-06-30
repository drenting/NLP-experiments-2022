This README is adapted from https://github.com/munikarmanish/bert-sentiment  
It was originally used in 'Fine-grained Sentiment Classification using BERT' (https://arxiv.org/abs/1910.03474).
This code trains a bert classifier for sentiment analysis and makes predictions on test data.

To install the required packages:
    pip install -r requirements.txt

To use the code: 
    run.py [OPTIONS]

  Options:
    -c, --bert-config TEXT  Pretrained BERT configuration (bert-large-uncased or bert-base-uncased)
    -b, --binary            Use binary labels, ignore neutrals, also predict on airline data
    -r, --root              Use only root nodes of SST
    -s, --save              Save the model files after every epoch
    -h, --help              Show this message and exit.

Commands used for the present study:
For the binary classifier, using BERT-base and only root nodes:
    run.py -c bert-base-uncased -rb

For the fine-grained classifier, using BERT-base and only root nodes:
    run.py -c bert-base-uncased -r
