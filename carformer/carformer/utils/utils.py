import torch
import numpy as np
from enum import IntEnum
from collections import defaultdict
from collections.abc import MutableMapping
import random


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class TokenTypeIDs(IntEnum):
    GOAL = 4
    STATE = 0
    BEV = 1
    ACTION = 2
    REWARD = 3
    EOS = -1


def deinterleave(interleaved_tensors, interleaved_token_type_ids, axis=1):
    """
    Deinterleave tensors along the specified axis.
    Args:
        interleaved_tensors: A tensor of shape (..., step_width, ...) to deinterleave
        interleaved_token_type_ids: A tensor of shape (batch, step_width) containing the token type ids
        axis: The axis to deinterleave along
    Returns:
        A dictionary of deinterleaved tensors, keyed by token type id
    """
    results_by_tokentype = defaultdict(list)

    for tokenType in TokenTypeIDs:
        max_axis_length = 0
        for batch_idx in range(interleaved_token_type_ids.shape[0]):
            # Get the indices of the unique token type ids
            token_type_id_indices = torch.where(
                interleaved_token_type_ids[batch_idx] == tokenType
            )[0]

            # Get the tensor corresponding to the token type id
            results_by_tokentype[tokenType].append(
                torch.index_select(
                    interleaved_tensors[batch_idx].unsqueeze(0),
                    axis,
                    token_type_id_indices,
                )
            )
            max_axis_length = max(
                max_axis_length, results_by_tokentype[tokenType][-1].shape[axis]
            )

        # Pad the tensors to the max length
        output_shape = list(results_by_tokentype[tokenType][0].shape)
        for i in range(len(results_by_tokentype[tokenType])):
            output_shape[axis] = (
                max_axis_length - results_by_tokentype[tokenType][i].shape[axis]
            )
            if output_shape[axis] > 0:
                results_by_tokentype[tokenType][i] = torch.cat(
                    [
                        results_by_tokentype[tokenType][i],
                        torch.zeros(
                            output_shape, dtype=results_by_tokentype[tokenType][i].dtype
                        ).to(results_by_tokentype[tokenType][i].device),
                    ],
                    axis=axis,
                )

        # Concatenate the tensors
        results_by_tokentype[tokenType] = torch.cat(
            results_by_tokentype[tokenType], axis=0
        )

    return results_by_tokentype


# Normalize version compatible with torch tensors
def normalize_angle_torch(x):
    x = x % (2 * np.pi)  # force in range [0, 2 pi)
    x = torch.where(x > np.pi, x - 2 * np.pi, x)  # move to [-pi, pi)
    return x


def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    else:
        return model


# Flatten backbone config nested dicts into a single dict
# a: {b: c} -> {"a.b": c}
def flatten_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def fuzzy_extract_state_dict_from_checkpoint(checkpoint, consume="ponderer."):
    if "model" in checkpoint:
        checkpoint = checkpoint["model"]

    if consume:  # Remove the prefix from the keys
        checkpoint = {
            (k[len(consume) :] if k.startswith(consume) else k): v
            for k, v in checkpoint.items()
        }

    return checkpoint
