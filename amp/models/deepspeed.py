import os

import deepspeed
import torch
from deepspeed.ops.op_builder import (WARNING, cuda_minor_mismatch_ok,
                                      installed_cuda_version)

from ..module_patcher import when_imported


def assert_no_cuda_mismatch_doesnt_check(name=""):
    cuda_major, cuda_minor = installed_cuda_version(name)
    sys_cuda_version = f'{cuda_major}.{cuda_minor}'
    torch_cuda_version = ".".join(torch.version.cuda.split('.')[:2])
    # This is a show-stopping error, should probably not proceed past this
    if sys_cuda_version != torch_cuda_version:
        if (cuda_major in cuda_minor_mismatch_ok and sys_cuda_version in cuda_minor_mismatch_ok[
                cuda_major] and torch_cuda_version in cuda_minor_mismatch_ok[cuda_major]):
            print(
                f"Installed CUDA version {sys_cuda_version} does not match the "
                f"version torch was compiled with {torch.version.cuda} "
                "but since the APIs are compatible, accepting this combination")
            return True
        elif os.getenv("DS_SKIP_CUDA_CHECK", "0") == "1":
            print(
                f"{WARNING} DeepSpeed Op Builder: Installed CUDA version {sys_cuda_version} does not match the "
                f"version torch was compiled with {torch.version.cuda}."
                "Detected `DS_SKIP_CUDA_CHECK=1`: Allowing this combination of CUDA, but it may result in unexpected behavior.")
            return True
        # raise CUDAMismatchException(
        #     f">- DeepSpeed Op Builder: Installed CUDA version {sys_cuda_version} does not match the "
        #     f"version torch was compiled with {torch.version.cuda}, unable to compile "
        #     "cuda/cpp extensions without a matching cuda version.")
    return True


@when_imported('deepspeed')
def init_deepspeed(mod):
    mod.ops.op_builder.assert_no_cuda_mismatch_doesnt = assert_no_cuda_mismatch_doesnt_check
