
import unittest
from collections import Counter
from fusanet_utils.datasets.external import ESC, UrbanSound8K

class test_external_datasets(unittest.TestCase):
    """
    TODO: PATH to Datasets!
    """

    def test_ESC(self):
        dataset = ESC('./datasets')
        self.assertEqual(Counter(dataset.labels)['animal/dog'], 40)

    def test_urbansound(self):
        dataset = UrbanSound8K('./datasets')
        self.assertEqual(Counter(dataset.labels)['animal/dog'], 1000)

if __name__ == '__main__':
    unittest.main()

