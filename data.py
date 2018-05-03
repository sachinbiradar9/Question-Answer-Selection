import random
from collections import namedtuple
import pickle


class Vocabulary(dict):
    """
    Bi-directional look up dictionary for the vocabulary

    Args:
        (dict): the default python dict class is extended
    """

    def __init__(self, vocabulary_file_name):
        with open(vocabulary_file_name) as vocabulary_file:
            for line in vocabulary_file:
                key, value = line.split()
                self[int(key)] = value
        self[0] = '<PAD>'

    def __setitem__(self, key, value):
        if key in self:
            raise Exception('Repeat Key', key)
        if value in self:
            raise Exception('Repeat value', value)
        dict.__setitem__(self, key, value)
        dict.__setitem__(self, value, key)

    def __delitem__(self, key):
        dict.__delitem__(self, self[key])
        dict.__delitem__(self, key)

    def __len__(self):
        return dict.__len__(self) // 2


class QAData():
    """
    Load the train/predecit/test data
    """

    def __init__(self):
        self.vocabulary = Vocabulary("./data/vocab_all.txt")
        self.dec_timesteps=150
        self.enc_timesteps=150
        self.answers = pickle.load(open("./data/answers.pkl",'rb'))
        self.training_set = pickle.load(open("./data/train.pkl",'rb'))

    def pad(self, data, length):
        """
        pad the data to meet given length requirement

        Args:
            data (vector): vector of question or answer
            length(integer): length of desired vector
        """

        from keras.preprocessing.sequence import pad_sequences
        return pad_sequences(data, maxlen=length, padding='post', truncating='post', value=0)

    def get_training_data(self):
        """
        Return training question and answers
        """

        questions = []
        good_answers = []
        for j, qa in enumerate(self.training_set):
            questions.extend([qa['question']] * len(qa['answers']))
            good_answers.extend([self.answers[i] for i in qa['answers']])

        # pad the question and answers
        questions = self.pad(questions, self.enc_timesteps)
        good_answers = self.pad(good_answers, self.dec_timesteps)
        bad_answers = self.pad(random.sample(list(self.answers.values()), len(good_answers)), self.dec_timesteps)

        return questions,good_answers,bad_answers

    def process_data(self, d):
        """
        Process the predection data
        """

        indices = d['good'] + d['bad']
        answers = self.pad([self.answers[i] for i in indices], self.dec_timesteps)
        question = self.pad([d['question']] * len(indices), self.enc_timesteps)
        return indices,answers,question

    def process_test_data(self, question, answers):
        """
        Process the test data
        """

        answer_unpadded = []
        for answer in answers:
            print (answer.split(' '))
            answer_unpadded.append([self.vocabulary[word] for word in answer.split(' ')])
        answers = self.pad(answer_unpadded, self.dec_timesteps)
        question = self.pad([[self.vocabulary[word] for word in question.split(' ')]] * len(answers), self.enc_timesteps)
        return answers, question
