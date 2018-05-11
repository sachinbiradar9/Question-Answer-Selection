# Question Answer Selection
Deep Learning framework to select best answer for a question from an answer candidate pool, it does not depend on manually defined features or linguistic tools. The basic framework was to build the embeddings of questions and answers based on bidirectional long short-term memory(biLSTM) models, and measure their closeness by cosine similarity.

<img src="/images/biLSTM.png" width="70%">

This basic biLSTM model was extended to define a more composite representation for questions and answers by combining convolutional neural network(CNN) on top of biLSTM.

<img src="/images/biLSTM-CNN.png" width="70%">

## Installation
`pip3 install numpy`  
`pip3 install keras`  
`pip3 install scipy`  
Dataset : [InsuranceQA Corpus](https://github.com/shuzi/insuranceQA)

## Usage
Run the server - `sudo python3 server.py`  
Open [localhost](http://localhost/)

## Credits
- [Dr. Plamen Petrov](http://business.uic.edu/faculty/plamen-petrov)
- Ming Tan, Bing Xiang and Bowen Zhou. 2016. [_LSTM-based Deep Learning Models for non-factoid answer selection_](http://arxiv.org/abs/1511.04108)
- Minwei Feng, Bing Xiang, Michael R. Glass, Lidan Wang, Bowen Zhou. 2015. [Applying Deep Learning to Answer Selection: A Study and An Open Task](https://arxiv.org/abs/1508.01585)
