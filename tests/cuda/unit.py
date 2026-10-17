"""Focused hardware checks for the installed CUDA buffer contract."""

import torch

from meshtrain.cuda_adapter import CudaStagingAdapter


def rejected(operation, kind=(TypeError, ValueError)):
    try:
        operation()
    except kind:
        return
    raise AssertionError("unsupported input was accepted")


def run(communicator):
    evidence = []
    for arguments in (dict(count=True), dict(count=-1), dict(count=2, max_staging_bytes=4),
                      dict(count=1, max_staging_bytes=True), dict(count=1, pin_memory=1),
                      dict(count=1, device="cuda"), dict(count=1, device="cpu"),
                      dict(count=1, device=0), dict(count=1, device="cuda:9999"),
                      dict(count=1, device="cuda:256"), dict(count=1, device=torch.device("cuda:128"))):
        rejected(lambda: CudaStagingAdapter(**(dict(device="cuda:0") | arguments)))
    evidence.append("constructor_validation")
    adapter = CudaStagingAdapter(3, device="cuda:0")
    tensor = torch.arange(3, dtype=torch.float32, device="cuda:0")

    class Subclass(torch.Tensor):
        pass

    invalid = [None, tensor.cpu(), tensor.double(), tensor.half(), tensor.reshape(1, 3),
               tensor[:2], torch.ones(6, device="cuda:0")[::2], tensor.clone().requires_grad_(),
               torch.nn.Parameter(tensor.clone(), requires_grad=False), tensor.as_subclass(Subclass),
               torch._neg_view(tensor), tensor.to_sparse()]
    baseline = tensor.clone()
    sequence = communicator.next_sequence
    for value in invalid:
        rejected(lambda: adapter.all_reduce_sum_(value, communicator))
        assert communicator.next_sequence == sequence
    rejected(lambda: adapter.all_reduce_sum_(tensor, object()))
    assert torch.equal(tensor, baseline)
    evidence.append("input_rejection_before_native")
    with torch.inference_mode():
        inference_input = torch.ones(3, device="cuda:0")
        ordinary = CudaStagingAdapter(3, device="cuda:0")
        assert not torch.is_inference(ordinary._staging)
        assert torch.is_inference_mode_enabled()
    rejected(lambda: ordinary.all_reduce_sum_(inference_input, communicator))
    assert communicator.next_sequence == sequence
    ordinary.all_reduce_sum_(tensor, communicator)
    assert torch.equal(tensor, baseline)
    evidence.append("inference_input_and_workspace")
    with torch.autograd.forward_ad.dual_level():
        primal, tangent = tensor.clone(), torch.full_like(tensor, 3)
        dual = torch.autograd.forward_ad.make_dual(primal, tangent)
        for inference_mode in (False, True):
            sequence = communicator.next_sequence
            with torch.inference_mode(inference_mode):
                rejected(lambda: adapter.all_reduce_sum_(dual, communicator))
                assert torch.is_inference_mode_enabled() == inference_mode
            assert communicator.next_sequence == sequence
            unpacked = torch.autograd.forward_ad.unpack_dual(dual)
            assert torch.equal(unpacked.primal, baseline) and torch.all(unpacked.tangent == 3).item()
    evidence.append("dual_rejection_including_caller_inference")
    torch.cuda.synchronize()
    sequence = communicator.next_sequence
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rejected(lambda: adapter.all_reduce_sum_(tensor, communicator), ValueError)
    assert communicator.next_sequence == sequence and torch.equal(tensor, baseline)
    evidence.append("capture_rejection")
    for pinned in (False, True):
        for count in (0, 1, 1025):
            workspace = CudaStagingAdapter(count, device="cuda:0", pin_memory=pinned)
            backing = torch.full((count + 2,), -19.0, device="cuda:0")
            value = backing[1:count + 1]
            value.fill_(7)
            pointer = value.data_ptr()
            sequence = communicator.next_sequence
            timing = workspace.all_reduce_sum_(value, communicator)
            assert communicator.next_sequence == sequence + 1
            assert pointer == value.data_ptr() and torch.all(value == 7).item()
            assert backing[0].item() == -19 and backing[-1].item() == -19
            assert workspace._view.__array_interface__["data"][0] == workspace._staging.data_ptr() or count == 0
            if count:
                assert workspace._staging.is_pinned() == pinned
                workspace._view[0] = 11
                assert workspace._staging[0].item() == 11
            else:
                assert timing.bytes == 0 and timing.d2h_ms == timing.h2d_ms == 0
    evidence.append("empty_identity_subview_pinning_and_shared_storage")
    bits = torch.tensor([0x7fc00001, 0x7f800000, -8388608, -2147483648, 1], dtype=torch.int32)
    value = bits.view(torch.float32).to("cuda:0")
    CudaStagingAdapter(5, device="cuda:0").all_reduce_sum_(value, communicator)
    assert torch.equal(value.view(torch.int32).cpu(), bits)
    evidence.append("single_rank_special_bits")
    default = torch.cuda.current_stream()
    producer, consumer, reader = (torch.cuda.Stream() for _ in range(3))
    value = torch.empty(1025, device="cuda:0")
    producer.wait_stream(default)
    workspace = CudaStagingAdapter(1025, device="cuda:0")
    with torch.cuda.stream(producer):
        value.fill_(17)
        workspace.all_reduce_sum_(value, communicator)
        assert producer.query()
    assert torch.cuda.current_stream() == default
    with torch.cuda.stream(producer):
        value.fill_(23)
    consumer.wait_stream(producer)
    with torch.cuda.stream(consumer):
        workspace.all_reduce_sum_(value, communicator)
        assert consumer.query()
    with torch.cuda.stream(reader):
        observed = value.clone()
    reader.synchronize()
    assert torch.all(observed == 23).item() and torch.cuda.current_stream() == default
    evidence.append("nondefault_producer_ordered_consumer_and_completed_return")
    return evidence
