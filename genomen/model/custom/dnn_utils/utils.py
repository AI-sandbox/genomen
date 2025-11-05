from typing import Any

import torch as t


def cleanup_gpu(model: Any):
    if t.cuda.is_available():
        # Clear cache
        t.cuda.empty_cache()

        # Move model to CPU to free GPU memory
        model.cpu()

        # Force garbage collection
        import gc

        gc.collect()

        # Clear cache again after cleanup
        t.cuda.empty_cache()
