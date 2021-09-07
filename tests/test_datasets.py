
import unittest
from datasets import ESC, UrbanSound8K
from collections import Counter

class test_datasets(unittest.TestCase):

    def test_ESC(self):
        dataset = ESC('./datasets')
        self.assertEqual(Counter(dataset.labels)['animal/dog'], 40)

    def test_urbansound(self):
        dataset = UrbanSound8K('./datasets')
        self.assertEqual(Counter(dataset.labels)['animal/dog'], 1000)

if __name__ == '__main__':
    unittest.main()

