import unittest

import redshells


class _DummyModel(object):
    pass


class PredictionModelFactoryTest(unittest.TestCase):
    def test_register_and_create(self):
        redshells.factory.register_prediction_model('test', _DummyModel)
        model = redshells.factory.create_prediction_model('test')
        self.assertIsInstance(model, _DummyModel)

    def test_create_with_unregistered(self):
        with self.assertRaises(RuntimeError):
            redshells.factory.create_prediction_model('test')


if __name__ == '__main__':
    unittest.main()
