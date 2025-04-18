# SPDX-License-Identifier: Apache-2.0
"""A layer that samples the next tokens from the model's outputs."""
import itertools
import warnings
from dataclasses import dataclass
from importlib.util import find_spec
from math import inf
from typing import Dict, Iterator, List, Optional, Tuple, Union

import msgspec
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.model_executor.layers.utils import apply_penalties
from vllm.model_executor.sampling_metadata import (SamplingMetadata,
                                                   SamplingTensors,
                                                   SequenceGroupToSample)
from vllm.sampling_params import SamplingType
from vllm.sequence import (VLLM_INVALID_TOKEN_ID,
                           CompletionSequenceGroupOutput, Logprob,
                           PromptLogprobs, SampleLogprobs, SequenceOutput)
from vllm.spec_decode.metrics import SpecDecodeWorkerMetrics

    
ARBITRARY_TOKEN_ID = 48763

if envs.VLLM_USE_FLASHINFER_SAMPLER and find_spec("flashinfer"):
    import flashinfer.sampling
    # yapf: disable
    from flashinfer.sampling import (
        top_k_top_p_sampling_from_probs as flashinfer_top_k_top_p_sampling)

    # yapf: enable
else:
    flashinfer_top_k_top_p_sampling = None


def get_sampler() -> torch.nn.Module:
    if envs.VLLM_USE_V1:
        # Lazy import: the v1 package isn't distributed
        from vllm.v1.sample.sampler import Sampler as V1Sampler
        return V1Sampler()
    return Sampler()


# (num_token_ids, num_parent_ids) per sequence group.
SampleResultType = List[Tuple[List[int], List[int]]]

# Types of temporary data structures used for
# computing sample_result
SampleMetadataType = Dict[SamplingType, Tuple[List[int],
                                              List[SequenceGroupToSample]]]
MultinomialSamplesType = Dict[SamplingType, torch.Tensor]
SampleResultsDictType = Dict[int, Tuple[List[int], List[int]]]


# Encapsulates temporary data structures for computing
# sample_result.
#
# * For multi-step scheduling: must be returned
#   by `Sampler.forward()` and used later to compute the pythonized
#   sample_result
#
# * For single-step scheduling: consumed immediately
#   inside `Sampler.forward()` to compute pythonized sample_result.
@dataclass
class SampleResultArgsType:
    sample_metadata: SampleMetadataType
    multinomial_samples: MultinomialSamplesType
    sample_results_dict: SampleResultsDictType
    sampling_metadata: SamplingMetadata
    greedy_samples: Optional[torch.Tensor]


# Union of non-deferred (single-step scheduling)
# vs deferred (multi-step scheduling)
# sample result types
MaybeDeferredSampleResultType = Union[SampleResultType, SampleResultArgsType]

# Abbreviation of the _sample() return type
SampleReturnType = Tuple[MaybeDeferredSampleResultType, Optional[torch.Tensor]]


class SamplerOutput(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        array_like=True):  # type: ignore[call-arg]
    """For each sequence group, we generate a list of SequenceOutput object,
    each of which contains one possible candidate for the next token.

    This data structure implements methods, so it can be used like a list, but
    also has optional fields for device tensors.
    """

    outputs: List[CompletionSequenceGroupOutput]

    # On-device tensor containing probabilities of each token.
    sampled_token_probs: Optional[torch.Tensor] = None

    # On-device tensor containing the logprobs of each token.
    logprobs: Optional["torch.Tensor"] = None

    # Holds either (1) the pythonized sampler result (single-step scheduling)
    # or (2) what will be arguments for later deferred pythonization of the
    # sampler result (muliti-step scheduling)
    deferred_sample_results_args: Optional[SampleResultArgsType] = None

    # On-device tensor containing the sampled token ids.
    sampled_token_ids: Optional[torch.Tensor] = None
    # CPU tensor containing the sampled token ids. Used during multi-step to
    # return the sampled token ids from last rank to AsyncLLMEngine to be
    # 'broadcasted' to all other PP ranks for next step.
    sampled_token_ids_cpu: Optional[torch.Tensor] = None

    # Spec decode metrics populated by workers.
    spec_decode_worker_metrics: Optional[SpecDecodeWorkerMetrics] = None

    # Optional last hidden states from the model.
    hidden_states: Optional[torch.Tensor] = None

    # Optional prefill hidden states from the model
    # (used for models like EAGLE).
    prefill_hidden_states: Optional[torch.Tensor] = None

    # Time taken in the forward pass for this across all workers
    model_forward_time: Optional[float] = None

    # Time taken in the model execute function. This will include model forward,
    # block/sync across workers, cpu-gpu sync time and sampling time.
    model_execute_time: Optional[float] = None

    def __getitem__(self, idx: int) -> CompletionSequenceGroupOutput:
        return self.outputs[idx]

    def __setitem__(self, idx: int, value):
        self.outputs[idx] = value

    def __iter__(self) -> Iterator[CompletionSequenceGroupOutput]:
        return iter(self.outputs)

    def __len__(self):
        return len(self.outputs)

    def __eq__(self, other: object):
        return isinstance(other,
                          self.__class__) and self.outputs == other.outputs

    def __repr__(self) -> str:
        """Show the shape of a tensor instead of its values to reduce noise.
        """
        sampled_token_probs_repr = ("None" if self.sampled_token_probs is None
                                    else self.sampled_token_probs.shape)
        sampled_token_ids_repr = ("None" if self.sampled_token_ids is None else
                                  self.sampled_token_ids.shape)
        return (
            f"SamplerOutput(outputs={self.outputs}, "
            f"sampled_token_probs={sampled_token_probs_repr}, "
            f"sampled_token_ids={sampled_token_ids_repr}, "
            f"spec_decode_worker_metrics={self.spec_decode_worker_metrics})")


class Sampler(nn.Module):
    """Samples the next tokens from the model's outputs.

    This layer does the following:
    1. Discard the hidden states that are not used for sampling (i.e., all
        tokens except the final one in each prompt).
    2. Compute the logits for the next tokens.
    3. Apply presence, frequency and repetition penalties.
    4. Apply temperature scaling.
    5. Apply top-p and top-k truncation.
    6. Sample the next tokens.
    Here, each sequence group within the batch can have different sampling
    parameters (e.g., sampling method, temperature, top-p, top-k, etc.).

    The structure of the logits tensor is coupled with the seq_groups in
    sampling_metadata. Typically, each sequence in each seq_group has one row in
    logits for the next token to be sampled; however, for a seq_group with a
    prompt request with the prompt_logprobs sampling parameter, there are rows
    in logits for each token in the input prompt.
    """

    def __init__(self):
        super().__init__()

        print("_______INITIALIZING SAMPLER_______")

        # Whether or not the SamplerOutput should have on-device tensors
        # containing the sampled token ids and probabilities. This is used by
        # speculative decoding.
        self.include_gpu_probs_tensor = False
        self.should_modify_greedy_probs_inplace = False


    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        """
        Single-step scheduling:
        * Perform GPU-side sampling computation & compute
          GPU-side logprobs tensor
        * Pythonize sampling result & logprobs tensor

        Multi-step scheduling:
        * Perform GPU-side sampling computation & compute
          GPU-side logprobs tensor
        * Defer Pythonization of sampling result & logprobs
          tensor
        * Encapsulate arguments required for deferred Pythonization
          in the :class:`SamplerOutput` structure

        Args:
            logits: (num_tokens, vocab_size).
            sampling_metadata: Metadata for sampling.
        """
        # No deferring of GPU->CPU pythonization, we must do it immediately
        # (Only single-token sampling)
        if sampling_metadata.skip_sampler_cpu_output:
            raise Exception("Cannot handle option to skip sampler cpu output!") 
        
        # No support for on-device tensor gpu_probs_tensor, which is only used
        # for speculative decoding
        if self.include_gpu_probs_tensor:
            raise Exception("Do not include option to include_gpu_probs_tensor")
        on_device_tensors = None

        assert logits is not None
        _, vocab_size = logits.shape

        # sampling_tensors are some kind of tensor metadata used for 
        # decoding, this does not apply to our greedy-only decoding. 

        # Sample the next tokens. Our _sample() calls _sample_with_torch()
        # which simply spits out RANDOM_TOKEN_ID for each seqgroup. 
        maybe_deferred_sample_results, _ = _sample(
            sampling_metadata,
            include_gpu_probs_tensor=self.include_gpu_probs_tensor,
        )

        # build_sampler_output can be left untouched, is pretty much just a python
        # object constructor.
        return _build_sampler_output(
            maybe_deferred_sample_results, # result of mocked _sample()
            sampling_metadata)


def _sample_with_torch(
    sampling_metadata: SamplingMetadata,
    include_gpu_probs_tensor: bool,
) -> SampleReturnType:
    '''Torch-oriented _sample() implementation.

    Single-step scheduling:
    * Perform GPU-side sampling computation
    * Immediately Pythonize sampling result

    Multi-step scheduling:
    * Perform GPU-side sampling computation
    * Defer Pythonization & preserve GPU-side
      tensors required for Pythonization
    '''
    
    # No on-device tensor gpu_probs_tensor, which is only used for 
    # speculative decoding
    if include_gpu_probs_tensor:
        raise Exception("Sampler output should not have on-device tensors!")
    sampled_token_ids_tensor = None
    
    # No deferring of GPU->CPU pythonization, we must do it immediately
    # (Only single-token sampling)
    if sampling_metadata.skip_sampler_cpu_output:
        raise Exception("Cannot handle option to skip sampler cpu output!") 
    
    categorized_seq_group_ids: Dict[SamplingType, List[int]] = {
        t: []
        for t in SamplingType
    }

    for i, seq_group in enumerate(sampling_metadata.seq_groups):
        sampling_params = seq_group.sampling_params
        sampling_type = sampling_params.sampling_type
        categorized_seq_group_ids[sampling_type].append(i)

        # DEBUG: print sampling type
        # print(f"SAMPLING TYPE: {sampling_type}")

    sample_results_dict: SampleResultsDictType = {}
    sample_metadata: SampleMetadataType = {}

    # Just assume that we only do greedy sampling. This can be set with temperature=0.0
    # Greedy sampling lets us get away with easy to parse seq_groups which is useful for our
    # profiling.
    sampling_type = SamplingType.GREEDY
    seq_group_id = categorized_seq_group_ids[sampling_type]
    seq_groups = [sampling_metadata.seq_groups[i] for i in seq_group_id]
    sample_metadata[sampling_type] = (seq_group_id, seq_groups)

    # We can skip the get_pythonized_sample results 
    # and directly unwrap the _greedy_sample method.
    # unwrapped the greedy_sample. We are allowed to simply iterate 
    # through seq_groups because seq_groups for greedy samples only 
    # contain 1. 
    sample_idx = 0
    sample_results: SampleResultType = []

    for seq_group in seq_groups:
        if not seq_group.do_sample:
            sample_results.append(([], []))
            continue 
        
        sample_results.append(([[ARBITRARY_TOKEN_ID], [0]]))
        sample_idx += 1
    
    # return the pythonization results
    sample_results_dict.update(zip(seq_group_id, sample_results))
    return [
        sample_results_dict.get(i, ([], []))
        for i in range(len(sampling_metadata.seq_groups))
    ], sampled_token_ids_tensor
 

def _sample(
    sampling_metadata: SamplingMetadata,
    include_gpu_probs_tensor: bool,
) -> SampleReturnType:
    """
    Args:
        probs: (num_query_tokens_in_batch, num_vocab)
        logprobs: (num_query_tokens_in_batch, num_vocab)
        sampling_metadata: The metadata for a batch for sampling.
        sampling_tensors: Tensors that include sampling related metadata.

    Returns:
        (next_token_ids, parent_seq_ids) for each seq group in a batch.
            If sampling is skipped, it returns ([], [])
        sampled_token_ids_tensor: A tensor of sampled token ids.
    """
    return _sample_with_torch(
        sampling_metadata,
        include_gpu_probs_tensor=include_gpu_probs_tensor,
    )


def _get_ranks(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    This function calculates the ranks of the chosen tokens in a logprob tensor.

    Args:
        x (torch.Tensor): 2D logprob tensor of shape (N, M)
                        where N is the no. of tokens and M is the vocab dim.
        indices (torch.Tensor): List of chosen token indices.

    Returns:
        torch.Tensor: 1D tensor of shape (N,) where N is the no. of tokens.
                    Each element in the returned tensor represents the rank
                    of the chosen token in the input logprob tensor.
    """
    vals = x[torch.arange(0, len(x), device=x.device, dtype=indices.dtype),
             indices]
    result = (x > vals[:, None])
    del vals
    return result.sum(1).add_(1)


def get_logprobs(
    logprobs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    sample_results: SampleResultType,
) -> Tuple[List[Optional[PromptLogprobs]], List[SampleLogprobs]]:
    """Return sample logprobs and prompt logprobs.

    The logic consists of 3 parts.
    - Select indices to compute logprob from, ranks of token ids, and
        the top k token ids from logprobs.
    - Compute prompt logprobs if required.
    - Compute sample logprobs if required.

    Args:
        logprobs: (num_query_tokens_across_batch, num_vocab). Each query token's
            logprob per vocab. Sequence groups' query tokens are batched in a
            single flattened tensor. For example, assuming there are N
            seq groups, it is sorted by prefill tokens for seq_group_1 (if
            prompt logprob is enabled), decode tokens for seq_group_1 (if
            sampling is required), prefill tokens for seq_group_2, ...
        sampling_metadata: The sampling metadata.
        sample_results: (num_seq_groups) The tuple of (next_token_ids,
            parent_ids) for each sequence group. When beam search is enabled,
            sample_results can contain different number of seq_ids from
            sampling_metadata.seq_groups. It is because beam search creates
            2 * BEAM_WIDTH number of samples (whereas there are only up to
            BEAM_WIDTH number of seq_ids).

    Returns:
        A tuple of prompt and sample logprobs per sequence group in a batch.
    """
    # The index of query token to calculate logprobs. It includes both
    # prompt and sample logprob indices.
    query_indices: List[int] = []
    # The next token ids to get the logprob value from.
    next_token_ids: List[int] = []
    # The largest requested number of logprobs. We find logprobs as many as the
    # largest num logprobs in this API. If every logprobs is None, it will be
    # set to -1.
    largest_num_logprobs = -1

    # We can have largest_num_logprobs =-1 if all logprobs is None

    # Select indices to compute logprob from, ranks of token ids, and the top
    # k token ids from logprobs.
    for (seq_group, sample_result) in zip(sampling_metadata.seq_groups,
                                          sample_results):
        sampling_params = seq_group.sampling_params

        # Update indices and tokens for prompt logprobs.
        if (seq_group.is_prompt
                and sampling_params.prompt_logprobs is not None):
            largest_num_logprobs = max(largest_num_logprobs,
                                       sampling_params.prompt_logprobs)
            next_prompt_tokens = _get_next_prompt_tokens(seq_group)
            query_indices.extend(seq_group.prompt_logprob_indices)
            next_token_ids.extend(next_prompt_tokens)

        # Update indices and next tokenes for sample logprob.
        if seq_group.do_sample:
            token_ids, parent_seq_ids = sample_result
            # NOTE: We cannot directly use sample_indices because
            # sample_indices only contain parent seq_ids of a previous step.
            # The current step may have different number of seq_ids, and
            # we can obtain it from `sample_result[1]`.
            query_idx = seq_group.sample_indices[0]
            query_indices.extend(
                [query_idx + parent_id for parent_id in parent_seq_ids])
            next_token_ids.extend(token_ids)

            if sampling_params.logprobs is not None:
                largest_num_logprobs = max(largest_num_logprobs,
                                           sampling_params.logprobs)

        assert len(next_token_ids) == len(query_indices)

    if len(query_indices) == 0:
        empty_sampled_logprob: SampleLogprobs = []
        empty_prompt_logprob: Optional[PromptLogprobs] = None
        num_seq_groups = len(sampling_metadata.seq_groups)
        return [empty_prompt_logprob
                ] * num_seq_groups, [empty_sampled_logprob] * num_seq_groups

    selected_logprobs, ranks = None, None
    top_logprobs, top_token_ids = None, None

    # If largest_num_logprobs == -1, i.e. no logprobs are requested, we can
    # skip the whole logprob calculation.
    if largest_num_logprobs >= 0:
        query_indices_gpu = torch.tensor(query_indices, device=logprobs.device)
        next_token_ids_gpu = torch.tensor(next_token_ids,
                                          device=logprobs.device)

        # (num_selected_query_tokens, num_logprobs). Note that query_indices can
        # contain duplicates if beam search is enabled.
        selected_logprobs = logprobs[[
            query_indices_gpu,
            next_token_ids_gpu,
        ]]
        ranks = _get_ranks(
            logprobs[query_indices_gpu],
            next_token_ids_gpu,
        )
        assert selected_logprobs.shape[0] == ranks.shape[0]

        # We need to compute top k only if there exists logprobs > 0.
        if largest_num_logprobs > 0:
            # Logprobs of topk tokens for a batch of sequence groups.
            # (num_query_tokens_across_batch).
            top_logprobs, top_token_ids = torch.topk(logprobs,
                                                     largest_num_logprobs,
                                                     dim=-1)
            top_logprobs = top_logprobs.to('cpu')
            top_token_ids = top_token_ids.to('cpu')

        selected_logprobs = selected_logprobs.to('cpu')
        ranks = ranks.to('cpu')

    # Find prompt/sample logprobs.
    prompt_logprobs_per_seq_group: List[Optional[PromptLogprobs]] = []
    sample_logprobs_per_seq_group: List[SampleLogprobs] = []
    top_logprob_idx = 0
    selected_logprobs_idx = 0

    for seq_group, sample_result in zip(sampling_metadata.seq_groups,
                                        sample_results):
        (prompt_logprobs, top_logprob_idx,
         selected_logprobs_idx) = _get_prompt_logprob_if_needed(
             seq_group, selected_logprobs, ranks, top_token_ids, top_logprobs,
             selected_logprobs_idx, top_logprob_idx)
        prompt_logprobs_per_seq_group.append(prompt_logprobs)

        (sampled_logprobs, top_logprob_idx,
         selected_logprobs_idx) = _get_sampled_logprob_if_needed(
             seq_group, sample_result, selected_logprobs, ranks, top_token_ids,
             top_logprobs, selected_logprobs_idx, top_logprob_idx)
        sample_logprobs_per_seq_group.append(sampled_logprobs)

    return prompt_logprobs_per_seq_group, sample_logprobs_per_seq_group


def _get_prompt_logprob_if_needed(
    seq_group: SequenceGroupToSample,
    selected_logprobs: torch.Tensor,
    ranks: torch.Tensor,
    top_token_ids: torch.Tensor,
    top_logprobs: torch.Tensor,
    selected_logprobs_idx: int,
    top_logprob_idx: int,
):
    """Compute the prompt logprob from a sequence group if needed."""
    sampling_params = seq_group.sampling_params
    is_prompt = seq_group.is_prompt

    # Find prompt logprobs
    prompt_logprobs: Optional[PromptLogprobs] = None
    if is_prompt and sampling_params.prompt_logprobs is not None:
        prompt_logprobs = []
        num_logprobs = sampling_params.prompt_logprobs
        next_prompt_tokens = _get_next_prompt_tokens(seq_group)
        # Pre-select indexes and create a list. It is faster than calling .item
        # repetitively.
        selected_logprob_items = selected_logprobs[
            selected_logprobs_idx:selected_logprobs_idx +
            len(next_prompt_tokens)].tolist()
        rank_items = ranks[selected_logprobs_idx:selected_logprobs_idx +
                           len(next_prompt_tokens)].tolist()

        for idx, token_id in enumerate(next_prompt_tokens):
            # Calculate the prompt logprob of the real prompt tokens.
            # {token_id: (logprob, rank_from_vocab)}
            prompt_logprobs_dict: Dict[int, Tuple[float, int]] = {
                token_id: (selected_logprob_items[idx], rank_items[idx])
            }

            # Add top K prompt logprobs along with its rank.
            if num_logprobs > 0:
                top_ids = top_token_ids[
                    top_logprob_idx, :num_logprobs].tolist()
                top_probs = top_logprobs[
                    top_logprob_idx, :num_logprobs].tolist()
                # Top K is already sorted by rank, so we can use 1 ~
                # num_logprobs + 1 for rank.
                top_ranks = range(1, num_logprobs + 1)
                prompt_logprobs_dict.update({
                    top_id: (top_prob, rank)
                    for top_id, top_prob, rank in zip(top_ids, top_probs,
                                                      top_ranks)
                })
            prompt_logprobs.append({
                token_id: Logprob(*logprob_and_rank)
                for token_id, logprob_and_rank in prompt_logprobs_dict.items()
            })
            # + 1 to go to the next prompt token.
            top_logprob_idx += 1

        # + len(next_prompt_tokens) to go to the next prompt.
        selected_logprobs_idx += len(next_prompt_tokens)
    return prompt_logprobs, top_logprob_idx, selected_logprobs_idx


def _get_sampled_logprob_if_needed(
    seq_group: SequenceGroupToSample,
    sample_result: Tuple[List[int], List[int]],
    selected_logprobs: torch.Tensor,
    ranks: torch.Tensor,
    top_token_ids: torch.Tensor,
    top_logprobs: torch.Tensor,
    selected_logprobs_idx: int,
    top_logprob_idx: int,
):
    """Compute the sample logprob if needed."""
    seq_ids = seq_group.seq_ids
    num_logprobs = seq_group.sampling_params.logprobs
    sampled_logprobs: SampleLogprobs = []
    next_token_ids, parent_seq_ids = sample_result

    if seq_group.do_sample:
        assert len(next_token_ids) > 0
        if num_logprobs is None:
            for next_token_id in next_token_ids:
                # Use a dummy logprob
                sampled_logprobs.append({next_token_id: Logprob(inf)})
        else:
            # Pre-select items from tensor. tolist() is faster than repetitive
            # `.item()` calls.
            selected_logprob_items = selected_logprobs[
                selected_logprobs_idx:selected_logprobs_idx +
                len(next_token_ids)].tolist()
            rank_items = ranks[selected_logprobs_idx:selected_logprobs_idx +
                               len(next_token_ids)].tolist()
            for idx, (next_token_id, parent_id) in enumerate(
                    zip(next_token_ids, parent_seq_ids)):
                # Get the logprob of a sampled token.
                sampled_logprobs_dict = {
                    next_token_id:
                    (selected_logprob_items[idx], rank_items[idx])
                }
                if num_logprobs is not None and num_logprobs > 0:
                    # Get top K logprobs.
                    top_ids = top_token_ids[top_logprob_idx +
                                            parent_id, :num_logprobs].tolist()
                    top_probs = top_logprobs[
                        top_logprob_idx + parent_id, :num_logprobs].tolist()
                    # Top K is already sorted by rank, so we can use 1 ~
                    # num_logprobs + 1 for rank.
                    top_ranks = range(1, num_logprobs + 1)
                    sampled_logprobs_dict.update({
                        top_id: (top_prob, rank)
                        for top_id, top_prob, rank in zip(
                            top_ids, top_probs, top_ranks)
                    })

                sampled_logprobs.append({
                    token_id: Logprob(*logprob_and_rank)
                    for token_id, logprob_and_rank in
                    sampled_logprobs_dict.items()
                })

        # NOTE: This part of code is not intuitive. `selected_logprobs` include
        # logprobs for the current step, which has len(next_token_ids) tokens
        # per sequence group. `logprobs` includes logprobs from the previous
        # steps, which has len(seq_ids) tokens per sequence group.

        # Iterate to the next sequence group in a batch.
        selected_logprobs_idx += len(next_token_ids)
        # Iterate to the next sequence group in a batch.
        top_logprob_idx += len(seq_ids)
    return sampled_logprobs, top_logprob_idx, selected_logprobs_idx


def _build_sampler_output(
    maybe_deferred_sample_results: MaybeDeferredSampleResultType,
    sampling_metadata: SamplingMetadata,
    skip_sampler_cpu_output: bool = False,
) -> SamplerOutput:
    """Construct Python objects with the output of sampling.

    Args:
        on_device_tensors: Tuple containing on-device tensors with the
            probabilities used in sampling and the sampled token ids. This
            allows post-processing without copies to CPU/serialization, e.g. in
            speculative decoding rejection sampling.
    """
    sampler_output: List[CompletionSequenceGroupOutput] = []

    if skip_sampler_cpu_output:
        raise Exception("We cannot skip cpu output!")
    assert not isinstance(maybe_deferred_sample_results,
                            SampleResultArgsType)
    assert len(sampling_metadata.seq_groups) \
        == len(maybe_deferred_sample_results)
    deferred_sample_results_args = None

    for (seq_group, sample_result) in zip(sampling_metadata.seq_groups,
                                        maybe_deferred_sample_results):
        seq_ids = seq_group.seq_ids
        next_token_ids, parent_ids = sample_result
        seq_outputs: List[SequenceOutput] = []
        for parent_id, next_token_id in zip(
                parent_ids, next_token_ids):
            seq_outputs.append(
                # SequenceOutput(seq_ids[parent_id], next_token_id,
                #                 logprobs))
                SequenceOutput(seq_ids[parent_id], next_token_id, None))
        sampler_output.append(
            # CompletionSequenceGroupOutput(seq_outputs,
            #                                 group_prompt_logprobs))
            CompletionSequenceGroupOutput(seq_outputs, None))

    # If not specified, store None values in SamplerOutput.
    sampled_token_probs, logprobs_tensor, sampled_token_ids = (None, None, None)

    return SamplerOutput(
        outputs=sampler_output,
        sampled_token_probs=sampled_token_probs,
        sampled_token_ids=sampled_token_ids,
        logprobs=logprobs_tensor,
        deferred_sample_results_args=deferred_sample_results_args)


def _get_next_prompt_tokens(
        seq_group: SequenceGroupToSample) -> tuple[int, ...]:
    """Get a list of next prompt tokens to compute logprob from a
        given sequence group.

    It is used to compute prompt logprob. Imagine you have logprob for each
    query token. Query token needs to know the next prompt token id to compute
    prompt logprob. This is a helper to obtain next prompt token ids.

    This API has to be used only when the caller knows seq_group is in prefill
    stage.

    Returns:
        A list of next prompt tokens to compute logprob.
    """
    assert seq_group.is_prompt, (
        "Caller should ensure the sequence group is in a prefill stage.")
    seq_ids = seq_group.seq_ids
    query_len = seq_group.query_len
    assert query_len is not None
    # prompt has only 1 seq id.
    assert len(seq_ids) == 1
    seq_data = seq_group.seq_data[seq_ids[0]]
    computed_len = seq_data.get_num_computed_tokens()
    prompt_tokens = seq_data.prompt_token_ids
    # +1 because we are looking for a next prompt token.
    next_token_index_start = computed_len + 1
    next_token_index_end = min(computed_len + query_len + 1,
                               len(prompt_tokens))
    next_prompt_tokens = prompt_tokens[
        next_token_index_start:next_token_index_end]
    return next_prompt_tokens