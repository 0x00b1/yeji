from typing import Dict, Sequence, Union

import torch
from torch import Tensor


def _protein_property_compute(
    scores: Tensor,
    count: Union[int, Tensor],
) -> Tensor:
    return torch.sum(scores) / count


def _protein_property_update(
    predictions: str | Sequence[str],
    index: Dict[str, float],
) -> Tensor:
    if isinstance(predictions, str):
        predictions = [predictions]

    scores = torch.zeros([len(predictions)])

    for prediction_index, prediction in enumerate(predictions):
        score = torch.tensor(0.0)

        for residue in prediction:
            score = score + index.get(residue, 0.0)

        scores[prediction_index] = score / len(prediction)

    return scores


def _protein_property(
    predictions: str | Sequence[str],
    index: Dict[str, float],
) -> Tensor:
    scores = _protein_property_update(predictions, index)

    return _protein_property_compute(scores, scores.shape[0])
