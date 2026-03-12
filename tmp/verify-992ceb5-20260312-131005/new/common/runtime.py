from contextlib import AbstractContextManager, nullcontext

import torch


def configure_torch_runtime(device: torch.device) -> None:
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def milo_autocast(device: torch.device) -> AbstractContextManager[object]:
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def is_cuda_out_of_memory(exc: RuntimeError) -> bool:
    return "out of memory" in str(exc).lower()
