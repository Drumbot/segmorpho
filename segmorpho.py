# !/usr/bin/env python
#  encoding: utf-8


import csv
import re
from pprint import pprint
import sklearn_crfsuite
# from sklearn_crfsuite import metrics
from sklearn import cross_validation, metrics
import random
from itertools import chain
import operator
from sklearn_crfsuite.utils import flatten

class DatasetParser(object):

    def __init__(self, data_filepath):
        self.data_file = data_filepath
        return

    def open(self):
        self.FILE = open(self.data_filepath)
        return

    def close(self):
        self.FILE.close()
        self.FILE = None
        return


class TrainDatasetParser(DatasetParser):
    """
    Responsible for formatting the CSV dataset into dicts of two elements:
    1) a character list for a given word
    2) a corresponding BMS label list

    example
    input : prévision  --  <pré<{vis>ion>}
    output: (['<s>',   'p', 'r', 'é', 'v', 'i', 's', 'i', 'o', 'n', '</s>'],
             ['START', 'B', 'M', 'M', 'B', 'M', 'M', 'B', 'M', 'M', 'STOP'])

    The dataset is expected to be a csv file with column headers
    'word', 'pos', and 'segmentation'.
    """

    def __init__(self):
        return

    


class DataManager(object):
    """
    Responsible for dataset parsing, instance labeling and feature extraction.
    """

    def __init__(self, max_substring_len):
        self.max_substring_len = max_substring_len
        return

    def observation_to_feature_sets(self, observation):
        """
        observation is a list of characters with first element = <s> and last
        element = </s>
        example: 'prévision'
        ['<s>', 'p', 'r', 'é', 'v', 'i', 's', 'i', 'o', 'n', '</s>']
        """
        obs_len = len(observation)
        feature_sets = []
        feature_sets.append({'BIAS': 1.0, 'right 1': '<s>'})

        for i in range(1, obs_len - 1):
            features = {'BIAS': 1.0}

            # extract left substring features
            for start in range(max(i - self.max_substring_len, 0), i):
                key = 'left {}'.format(i - start)
                features.update({key: ''.join(observation[start:i])})

            # extract right substring features
            for stop in range(i + 1, min(i + self.max_substring_len, obs_len) + 1):
                key = 'right {}'.format(stop - i)
                features.update({key: ''.join(observation[i:stop])})
            feature_sets.append(features)
        feature_sets.append({'BIAS': 1.0, 'right 1': '</s>'})
        return feature_sets

    def process_dataset(self, data_filepath):

        x_sequences = []
        y_sequences = []
        count = 0

        with open(data_filepath, 'r') as f:
            data = csv.DictReader(f)

            for row in data:
                observation = ['<s>'] + list(row['word'].replace('-', '')) + ['</s']

                x = self.observation_to_feature_sets(observation)
                y = self.segmentation_to_labels(row['segmentation'])
                if len(x) != len(y):
                    # print(len(x), len(y))
                    # print(row['word'])
                    # print(row['segmentation'])
                    # print(y)
                    continue
                if any([not isinstance(x, str) for x in y]):
                    print(y)
                x_sequences.append(x)
                y_sequences.append(y)
        print(count)
        return (x_sequences, y_sequences)

    # def segmentation_to_labels(self, segmentation):
    #     """
    #     Segmentations come in a custom format, described in documentation.
    #     This outputs the sequence of labels that correspond to a segmentation.
    #     """
    #     segmentation = re.sub(r'^[<{-]+', '', segmentation)
    #     segmentation = re.sub(r'[}>-]+$', '', segmentation)
    #     segmentation = re.split(r'[<>{}-]+', segmentation)

    #     labels = ['START']
    #     for morph in segmentation:
    #         labels += ['B'] + ['M' for char in morph[1:]]
    #     labels.append('STOP')

    #     # Change label to S for single character morphs
    #     for i in range(len(labels) - 1):
    #         if labels[i] == 'B' and labels[i+1] in ['B', 'STOP']:
    #             labels[i] = 'S'

    #     return labels

    def segmentation_to_labels(self, segmentation):
        """
        Segmentations come in a custom format, described in documentation.
        This outputs the sequence of labels that correspond to a segmentation.
        """
        sp = re.split(r'<+', segmentation)
        prefs = [x for x in sp[:-1] if re.match(r'\w+', x)]
        
        rest = sp[-1]
        sp = re.split(r'>+', rest)
        suffs = [x for x in sp[1:] if re.match(r'\w+', x)]
        
        rest = sp[0]
        roots = [x for x in re.split(r'\W+', rest) if re.match(r'\w+', x)]

        labels = ['START']
        for pref in prefs:
            labels += ['B_PREF'] + ['M_PREF' for char in pref[1:]]
        for root in roots:
            labels += ['B_ROOT'] + ['M_ROOT' for char in root[1:]]
        for suff in suffs:
            labels += ['B_SUFF'] + ['M_SUFF' for char in suff[1:]]
        labels.append('STOP')

        # Change label to S for single character morphs
        for i in range(len(labels) - 1):
            if labels[i].startswith('B') and labels[i+1][0] in ['B', 'S']:
                labels[i] = 'S' + labels[i][1:]

        return labels


def get_average_label_probability(marginal_probabilities):
    """
    Returns the average probability of each highest probability labels in a
    sequence.
    """
    highest_probabilities = []
    for x in marginal_probabilities[1:-1]:  # Don't include probs for START and STOP
        hp = max(x.items(), key=operator.itemgetter(1))[1]
        highest_probabilities.append(hp)

    return sum(highest_probabilities) / len(highest_probabilities)



if __name__ == '__main__':
    data_manager = DataManager(5)
    print("Processing dataset")
    x, y = data_manager.process_dataset('/data/morpho/FLP_segmented.csv')
    temp = list(zip(x, y))
    random.shuffle(temp)
    x[:], y[:] = zip(*temp)
    x_folds = [x[i:i+len(x)//5] for i in range(0, len(x), len(x)//5)]
    y_folds = [y[i:i+len(y)//5] for i in range(0, len(y), len(y)//5)]

    crf = sklearn_crfsuite.CRF()
    # scores = cross_validation.cross_val_score(crf, x, y, cv=10, scoring='f1')
    scores = []
    # crf.fit(x[:100], y[:100])
    # print(crf.predict_marginals(x))
    for i in range(5):
        x_test = flatten(x_folds[:i] + x_folds[i+1:])
        x_train = x_folds[i]
        y_test = flatten(y_folds[:i] + y_folds[i+1:])
        y_train = y_folds[i]
        print(len(x_train[0]), len(y_train[0]))

        
        crf.fit(x_train, y_train)
        y_pred = crf.predict(x_test)
        print(len(y_pred))
        y_pred = flatten(y_pred)
        # y_marg = crf.predict_marginals(x_test)
        y_test = flatten(y_test)
        score = metrics.classification_report(y_test, y_pred,
                                              labels=['B_PREF', 'M_PREF', 
                                                      'B_ROOT', 'M_ROOT', 
                                                      'B_SUFF', 'M_SUFF', 'S'])
        print(score)

        # probs_for_correct = []
        # probs_for_mistake = []
        # for pred, true, marg in zip(y_pred, y_test, y_marg):
        #     avg_prob = get_average_label_probability(marg)
        #     if pred == true:
        #         probs_for_correct.append(avg_prob)
        #     else:
        #         probs_for_mistake.append(avg_prob)
        # avg_correct = sum(probs_for_correct) / len(probs_for_correct)
        # avg_mistake = sum(probs_for_mistake) / len(probs_for_mistake)
        # print('Average highest marginal probability (correct): %.8f' % avg_correct)
        # print('Min avg marginal probability (correct): %.8f' % min(probs_for_correct))
        # print('Max avg marginal probability (correct): %.8f' % max(probs_for_correct))
        # print('Average highest marginal probability (mistake): %.8f' % avg_mistake)
        # print('Min avg marginal probability (mistake): %.8f' % min(probs_for_mistake))
        # print('Max avg marginal probability (mistake): %.8f' % max(probs_for_mistake))

        # print(score)
    #     scores.append(score)
    # print(sum(scores)/5)

