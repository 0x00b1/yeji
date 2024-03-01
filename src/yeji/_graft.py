import torch
import torch.nn.functional
from torch import LongTensor


def graft(
    input: LongTensor,
    mask: LongTensor,
    template: LongTensor,
    mixed: bool = False,
    num_classes: int = -1,
) -> LongTensor:
    """
    Parameters:
    -----------
    input : Tensor, shape=[..., L]
        Sequences.

    mask : Tensor, shape=[L]
        Whether to draw sequence columns from the input or from the template.

    template : Tensor, shape=[T, L]
        Library of template sequences that may be grafted with the `input`
        sequence. Zero columns are not covered by the template.

    mixed : bool
        If `True`, find the contiguous template segments from `mask` choosing a
        different template for each segment, generating >> T solutions for each
        input. If `False`, choose exactly one template to graft.

    Returns
    -------
    grafted_sequence : Tensor, (..., T, L, 20)
        Grafted sequences.
    """

    def fn(input: LongTensor, mask: LongTensor, template: LongTensor):
        input = torch.nn.functional.one_hot(input, num_classes=num_classes)

        template = torch.nn.functional.one_hot(
            template, num_classes=num_classes
        )

        if mixed:
            raise NotImplementedError()
        else:
            return torch.argmax(
                torch.subtract(
                    torch.multiply(
                        input,
                        torch.unsqueeze(mask, -1),
                    ),
                    torch.multiply(
                        template,
                        torch.subtract(
                            torch.unsqueeze(mask, -1),
                            1,
                        ),
                    ),
                ),
                dim=-1,
            )

    return torch.func.vmap(fn, in_dims=(0, None, None))(input, mask, template)


def test_graft():
    """
    test the tensor_grafting function

    construct a one-hot encoded sequence tensor, a set of templates, and a mask_indices tensor
    """

    # first construct a one-hot encoded sequence tensor
    input = torch.nn.functional.one_hot(
        torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [9, 9, 9, 9, 9, 9, 9],
            ]
        ),
        num_classes=20,
    )

    template = torch.stack(
        [
            torch.nn.functional.one_hot(torch.tensor([1] * 7), num_classes=20),
            torch.nn.functional.one_hot(torch.tensor([2] * 7), num_classes=20),
            torch.nn.functional.one_hot(torch.tensor([3] * 7), num_classes=20),
        ]
    )

    mask = torch.tensor([[1, 0, 1, 0, 1, 0, 1]])

    a = graft(input[0], mask[0], template)
    b = graft(input[1], mask[0], template)

    grafted = torch.stack([a, b])

    torch.testing.assert_close(
        grafted,
        torch.stack(
            [
                torch.stack(
                    [
                        torch.nn.functional.one_hot(
                            torch.tensor([0, 1, 0, 1, 0, 1, 0]), num_classes=20
                        ),
                        torch.nn.functional.one_hot(
                            torch.tensor([0, 2, 0, 2, 0, 2, 0]), num_classes=20
                        ),
                        torch.nn.functional.one_hot(
                            torch.tensor([0, 3, 0, 3, 0, 3, 0]), num_classes=20
                        ),
                    ]
                ),
                torch.stack(
                    [
                        torch.nn.functional.one_hot(
                            torch.tensor([9, 1, 9, 1, 9, 1, 9]), num_classes=20
                        ),
                        torch.nn.functional.one_hot(
                            torch.tensor([9, 2, 9, 2, 9, 2, 9]), num_classes=20
                        ),
                        torch.nn.functional.one_hot(
                            torch.tensor([9, 3, 9, 3, 9, 3, 9]), num_classes=20
                        ),
                    ]
                ),
            ]
        ),
    )


test_graft()
