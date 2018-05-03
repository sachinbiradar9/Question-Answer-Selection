import numpy as np
from model import QAModel
from data import QAData, Vocabulary
import pickle
import random
from scipy.stats import rankdata

def main(mode='test', question=None, answers=None):
    """
    This function is used to train, predict or test

    Args:
        mode (str): train/preddict/test
        question (str): this contains the question
        answers (list): this contains list of answers in string format

    Returns:
        index (integer): index of the most likely answer
    """

    # get the train and predict model model
    vocabulary = Vocabulary("./data/vocab_all.txt")
    embedding_file = "./data/word2vec_100_dim.embeddings"
    qa_model = QAModel()
    train_model, predict_model = qa_model.get_lstm_cnn_model(embedding_file, len(vocabulary))

    epoch = 1
    if mode == 'train':
        for i in range(epoch):
            print ('Training epoch', i)

            # load training data
            qa_data = QAData()
            questions, good_answers, bad_answers = qa_data.get_training_data()

            # train the model
            Y = np.zeros(shape=(questions.shape[0],))
            train_model.fit(
                [questions, good_answers, bad_answers],
                Y,
                epochs=1,
                batch_size=64,
                validation_split=0.1,
                verbose=1
            )

            # save the trained model
            train_model.save_weights('model/train_weights_epoch_' + str(epoch) + '.h5', overwrite=True)
            predict_model.save_weights('model/predict_weights_epoch_' + str(epoch) + '.h5', overwrite=True)
    elif mode == 'predict':
        # load the evaluation data
        data = pickle.load(open("./data/dev.pkl",'rb'))
        random.shuffle(data)

        # load weights from trained model
        qa_data = QAData()
        predict_model.load_weights('model/cnnlastm_predict_weights_epoch_1.h5')

        c = 0
        c1 = 0
        for i, d in enumerate(data):
            print (i, len(data))

            # pad the data and get it in desired format
            indices, answers, question = qa_data.process_data(d)

            # get the similarity score
            sims = predict_model.predict([question, answers])

            n_good = len(d['good'])
            max_r = np.argmax(sims)
            max_n = np.argmax(sims[:n_good])
            r = rankdata(sims, method='max')
            c += 1 if max_r == max_n else 0
            c1 += 1 / float(r[max_r] - r[max_n] + 1)

        precision = c / float(len(data))
        mrr = c1 / float(len(data))
        print ("Precision", precision)
        print ("MRR", mrr)
    elif mode == 'test':
        # question and answers come from params
        qa_data = QAData()
        answers, question = qa_data.process_test_data(question, answers)

        # load weights from the trained model
        predict_model.load_weights('model/cnnlastm_predict_weights_epoch_1.h5')

        # get similarity score
        sims = predict_model.predict([question, answers])
        max_r = np.argmax(sims)
        return max_r

if __name__ == "__main__":
    main(mode='predict')

def test(question, answers):
    return main(mode='test', question=question, answers=answers)
