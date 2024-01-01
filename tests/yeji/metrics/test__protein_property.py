import torch
from yeji.metrics._protein_property import _ProteinProperty


class TestProteinProperty:
    index = {"A": 1.0, "C": 2.0, "D": 3.0}

    def test___init__(self):
        metric = _ProteinProperty(self.index)

        assert isinstance(metric, _ProteinProperty)

        assert metric.index == self.index

        torch.testing.assert_close(metric.score, torch.tensor(0.0))

        torch.testing.assert_close(metric.count, torch.tensor(0))

    def test_update(self):
        metric = _ProteinProperty(self.index)

        metric.update("ACDAC")

        torch.testing.assert_close(metric.score, torch.tensor(1.8))

        torch.testing.assert_close(metric.count, torch.tensor(1))

        metric = _ProteinProperty(self.index)

        metric.update("ACD")
        metric.update("AC")

        torch.testing.assert_close(metric.score, torch.tensor(3.5))

        torch.testing.assert_close(metric.count, torch.tensor(2))

    def test_compute(self):
        metric = _ProteinProperty(self.index)

        metric.update("ACD")
        metric.update("AC")

        torch.testing.assert_close(metric.compute(), torch.tensor(1.75))

        metric = _ProteinProperty(self.index)

        metric.update("ACDXYZ")

        torch.testing.assert_close(metric.score, torch.tensor(1.0))

        torch.testing.assert_close(metric.count, torch.tensor(1))
