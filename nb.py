import pandas as pd
import numpy as np
from collections import defaultdict
import re
from sklearn.datasets import fetch_20newsgroups
from pprint import pprint

from sklearn.model_selection import train_test_split


def preprocess_string(str_arg):
    cleaned_str = re.sub('[^a-z\s]+', ' ', str_arg, flags=re.IGNORECASE)
    cleaned_str = re.sub('(\s+)', ' ', cleaned_str)
    cleaned_str = cleaned_str.lower()

    return cleaned_str


class NaiveBayes:

    def __init__(self, unique_classes):

        self.classes = unique_classes  # Constructor is sinply passed with unique number of classes of the training set

    def addToBow(self, example, dict_index):


        if isinstance(example, np.ndarray): example = example[0]

        for token_word in example.split():  # for every word in preprocessed example

            self.bow_dicts[dict_index][token_word] += 1  # increment in its count

    def train(self, dataset, labels):



        self.examples = dataset
        self.labels = labels
        self.bow_dicts = np.array([defaultdict(lambda: 0) for index in range(self.classes.shape[0])])

        # only convert to numpy arrays if initially not passed as numpy arrays - else its a useless recomputation

        if not isinstance(self.examples, np.ndarray): self.examples = np.array(self.examples)
        if not isinstance(self.labels, np.ndarray): self.labels = np.array(self.labels)

        # constructing BoW for each category
        for cat_index, cat in enumerate(self.classes):
            all_cat_examples = self.examples[self.labels == cat]  # filter all examples of category == cat

            # get examples preprocessed

            #cleaned_examples = [preprocess_string(cat_example) for cat_example in all_cat_examples]

            cleaned_examples = pd.DataFrame(data=all_cat_examples)

            # now costruct BoW of this particular category
            np.apply_along_axis(self.addToBow, 1, cleaned_examples, cat_index)

        ###################################################################################################



        ###################################################################################################

        prob_classes = np.empty(self.classes.shape[0])
        all_words = []
        cat_word_counts = np.empty(self.classes.shape[0])
        for cat_index, cat in enumerate(self.classes):
            # Calculating prior probability p(c) for each class
            prob_classes[cat_index] = np.sum(self.labels == cat) / float(self.labels.shape[0])

            # Calculating total counts of all the words of each class
            count = list(self.bow_dicts[cat_index].values())
            cat_word_counts[cat_index] = np.sum(
                np.array(list(self.bow_dicts[cat_index].values()))) + 1  # |v| is remaining to be added

            # get all words of this category
            all_words += self.bow_dicts[cat_index].keys()

        # combine all words of every category & make them unique to get vocabulary -V- of entire training set

        self.vocab = np.unique(np.array(all_words))
        self.vocab_length = self.vocab.shape[0]

        # computing denominator value
        denoms = np.array(
            [cat_word_counts[cat_index] + self.vocab_length + 1 for cat_index, cat in enumerate(self.classes)])



        self.cats_info = [(self.bow_dicts[cat_index], prob_classes[cat_index], denoms[cat_index]) for cat_index, cat in
                          enumerate(self.classes)]
        self.cats_info = np.array(self.cats_info)

    def getExampleProb(self, test_example):



        likelihood_prob = np.zeros(self.classes.shape[0])  # to store probability w.r.t each class

        # finding probability w.r.t each class of the given test example
        for cat_index, cat in enumerate(self.classes):

            for test_token in test_example.split():  # split the test example and get p of each test word

                ####################################################################################

                # This loop computes : for each word w [ count(w|c)+1 ] / [ count(c) + |V| + 1 ]

                ####################################################################################

                # get total count of this test token from it's respective training dict to get numerator value
                test_token_counts = self.cats_info[cat_index][0].get(test_token, 0) + 1

                # now get likelihood of this test_token word
                test_token_prob = test_token_counts / float(self.cats_info[cat_index][2])

                # remember why taking log? To prevent underflow!
                likelihood_prob[cat_index] += np.log(test_token_prob)

        # we have likelihood estimate of the given example against every class but we need posterior probility
        post_prob = np.empty(self.classes.shape[0])
        for cat_index, cat in enumerate(self.classes):
            post_prob[cat_index] = likelihood_prob[cat_index] + np.log(self.cats_info[cat_index][1])

        return post_prob

    def test(self, test_set):



        predictions = []  # to store prediction of each test example
        for example in test_set:
            # preprocess the test example the same way we did for training set exampels
            cleaned_example = preprocess_string(example)

            # simply get the posterior probability of every example
            post_prob = self.getExampleProb(cleaned_example)  # get prob of this example for both classes

            # simply pick the max value and map against self.classes!
            predictions.append(self.classes[np.argmax(post_prob)])

        return np.array(predictions)


data = pd.read_csv('data.csv')

x1 = data.drop('class', axis=1)
y1 = data['class']

x2 = np.array(x1)
y2 = np.array(y1)

X_train,X_test,y_train,y_test=train_test_split(x2,y2,test_size=0.2)

categories=['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
newsgroups_train=fetch_20newsgroups(subset='train',categories=categories)

train_data = newsgroups_train.data
train_labels = newsgroups_train.target

print("Totaln number of training examples ",len(train_data))
print("total number of training labels ", len(train_labels))

print ("------------------- Dataset Categories -------------- ")
pprint(list(newsgroups_train.target_names))

pd.options.display.max_colwidth=250
pd.DataFrame(data=np.column_stack([train_data,train_labels]),columns=["Training Examples","Training Labels"]).head()


nb = NaiveBayes(np.unique(y_train))

print("----------- Training in proces - ----------------")

nb.train(X_train, y_train)

print("--------------- Training completed ----------------------")



newsgroups_test = fetch_20newsgroups(subset="test", categories = categories)
test_data = newsgroups_test.data
test_labels = newsgroups_test.target
print("number of test examples ", len(test_data))
print("number of test labels ", len(test_labels))


pclasses = nb.test(X_test)

test_acc = np.sum(pclasses==y_test)/float(y_test.shape[0])

print("Test set examples ", test_labels.shape[0])
print("test set accuraty", test_acc*100,"%")
