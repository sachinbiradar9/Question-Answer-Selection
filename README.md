# Question-Answering-System
Dataset : [InsuranceQA Corpus](https://github.com/shuzi/insuranceQA)  
Paper : [LSTM-based Deep Learning Models for Non-factoid Answer Selection](https://arxiv.org/pdf/1511.04108v4.pdf)

Deep Learning framework to select best answer for a question from an answer candidate pool, it does not depend on manually defined features or linguistic tools. The basic framework was to build the embeddings of questions and answers based on bidirectional long short-term memory(biLSTM) models, and measure their closeness by cosine similarity.

![biLSTM model](/images/biLSTM.png)

This basic biLSTM model was extended to define a more composite representation for questions and answers by combining convolutional neural network(CNN) on top of biLSTM.

![biLSTM/CNN model](/images/biLSTM/CNN.png)
