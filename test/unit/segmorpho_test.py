# !/usr/bin/env python
#  encoding: utf-8

import sys
sys.path.append('/home/hugo/Projects/segmorpho')
import unittest
from segmorpho.utilities import DataHandler, TrainDatasetParser
from mock import mock_open, patch


# class TestTrainDatasetParser(unittest.TestCase):

#     def test_parse_returns_correct_result(self):
#         self.maxDiff = None
#         expected = [{'observation': ['<s>', 'a', 'b', 'b', 'r', 'e', 'v', 'i', 'a', 't', 'e', '</s>'], 
#                      'labels': ['START', 'B', 'M', 'B', 'M', 'M', 'M', 'M', 'B', 'M', 'M', 'STOP']}, 
#                     {'observation': ['<s>', 'a', 'b', 'l', 'a', 'z', 'e', '</s>'], 
#                      'labels': ['START', 'S', 'B', 'M', 'M', 'M', 'M', 'STOP']}]
#         parser = TrainDatasetParser()
#         test_dataset_filepath = '/home/hugo/Projects/segmorpho/segmorpho/test/unit/test_dataset.csv'
#         self.assertEqual(parser.parse(test_dataset_filepath), expected)


class TestDataManager(unittest.TestCase):

    def test_observation_to_feature_sets_returns_correct_result_len_3(self):
        data_handler = DataHandler(3)
        observation = ['<s>', 'd', 'é', 'c', 'l', 'o', 'u', 'e', 'r', '</s>']
        expected = {0: ['BIAS', 'right <s>'],
                    1: ['BIAS', 'left <s>', 'right d', 'right dé', 'right déc'],
                    2: ['BIAS', 'left <s>d', 'left d', 'right é', 'right éc', 'right écl'],
                    3: ['BIAS', 'left <s>dé', 'left dé', 'left é', 'right c', 'right cl', 'right clo'],
                    4: ['BIAS', 'left déc', 'left éc', 'left c', 'right l', 'right lo', 'right lou'],
                    5: ['BIAS', 'left écl', 'left cl', 'left l', 'right o', 'right ou', 'right oue'],
                    6: ['BIAS', 'left clo', 'left lo', 'left o', 'right u', 'right ue', 'right uer'],
                    7: ['BIAS', 'left lou', 'left ou', 'left u', 'right e', 'right er', 'right er</s>'],
                    8: ['BIAS', 'left oue', 'left ue', 'left e', 'right r', 'right r</s>'],
                    9: ['BIAS', 'right </s>'],
                   }
        self.assertEqual(data_handler.observation_to_feature_sets(observation), expected)

    def test_observation_to_feature_sets_returns_correct_result_len_5(self):
        data_handler = DataHandler(5)
        observation = ['<s>', 'd', 'é', 'c', 'l', 'o', 'u', 'e', 'r', '</s>']
        expected = {0: ['BIAS', 'right <s>'],
                    1: ['BIAS', 'left <s>', 'right d', 'right dé', 'right déc', 'right décl', 'right déclo'],
                    2: ['BIAS', 'left <s>d', 'left d', 'right é', 'right éc', 'right écl', 'right éclo', 'right éclou'],
                    3: ['BIAS', 'left <s>dé', 'left dé', 'left é', 'right c', 'right cl', 'right clo', 'right clou', 'right cloue'],
                    4: ['BIAS', 'left <s>déc', 'left déc', 'left éc', 'left c', 'right l', 'right lo', 'right lou', 'right loue', 'right louer'],
                    5: ['BIAS', 'left <s>décl', 'left décl', 'left écl', 'left cl', 'left l', 'right o', 'right ou', 'right oue', 'right ouer', 'right ouer</s>'],
                    6: ['BIAS', 'left déclo','left éclo', 'left clo', 'left lo', 'left o', 'right u', 'right ue', 'right uer', 'right uer</s>'],
                    7: ['BIAS', 'left éclou', 'left clou', 'left lou', 'left ou', 'left u', 'right e', 'right er', 'right er</s>'],
                    8: ['BIAS', 'left cloue', 'left loue', 'left oue', 'left ue', 'left e', 'right r', 'right r</s>'],
                    9: ['BIAS', 'right </s>'],
                   }
        self.assertEqual(data_handler.observation_to_feature_sets(observation), expected)

    def test_segmentation_to_labels_returns_correct_result_no_segmentation(self):
        data_handler = DataHandler(1)
        expected = ['START', 'B', 'M', 'M', 'M', 'M', 'STOP']
        self.assertEqual(data_handler.segmentation_to_labels())
if __name__ == '__main__':
    unittest.main()