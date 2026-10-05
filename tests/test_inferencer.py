import unittest
import numpy as np
from src.engine import Inferencer

class TestInferencer(unittest.TestCase):

    def setUp(self):

        self.inferencer = Inferencer(model_path='models/EdgeConv_best.pt')  
        self.raw_jet = np.zeros((200, 4), dtype=np.float32)
        self.raw_jet[0] = [10.0, 3.0, 4.0, 5.0] 
        self.raw_jet[1] = [8.0,  2.0, 3.0, 1.0]

    def test_predict_schema(self):
        """
            test that the output of the predict method has the expected schema and types
        """

        result = self.inferencer.predict(self.raw_jet)

        self.assertIn('score', result)
        self.assertIn('prediction', result)
        self.assertIn('latency_ms', result)
        self.assertIsInstance(result['score'], float)
        self.assertIsInstance(result['prediction'], int)
        self.assertIsInstance(result['latency_ms'], float)

    def test_score_range(self):
        """
            test that the score is between 0 and 1
        """
        result = self.inferencer.predict(self.raw_jet)
        self.assertGreaterEqual(result['score'], 0.0)
        self.assertLessEqual(result['score'], 1.0)

    def test_prediction_is_binary(self):
        """
            test that the prediction is either 0 or 1
        """
        result = self.inferencer.predict(self.raw_jet)
        self.assertIn(result['prediction'], [0, 1])

    def test_invalid_shape(self):
        """
            test that an invalid input shape raises a ValueError
        """
        invalid_jet = np.zeros((100, 4), dtype=np.float32)  # Invalid shape
        with self.assertRaises(ValueError):
            self.inferencer.predict(invalid_jet)

    def test_batch_throughput_prediction(self):
        """
            verify that the troughput is a valid number (positive)
        """

        batch = np.stack([self.raw_jet for _ in range(5)])  # Create a batch of 5 jets
        result = self.inferencer.predict_batch(batch)
        self.assertIn('throughput', result)
        self.assertIsInstance(result['throughput'], float)
        self.assertGreater(result['throughput'], 0.0)

    def test_batch_prediction_schema(self):
        """
            test that the output of the predict_batch method has the expected schema and types
        """
        for n in [1, 5, 20]:
            jets = np.stack([self.raw_jet for _ in range(n)])  # Create a batch of n jets
            result = self.inferencer.predict_batch(jets)
            self.assertEqual(len(result['results']), n)

    def test_model_not_found_raises(self):
        """
            test that an invalid model path raises a FileNotFoundError
        """
        with self.assertRaises(FileNotFoundError):
            Inferencer("invalid_model.pt")

    