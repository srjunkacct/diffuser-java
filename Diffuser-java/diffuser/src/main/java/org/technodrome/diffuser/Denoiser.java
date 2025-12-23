package org.technodrome.diffuser;

import ai.djl.ndarray.NDArray;

public interface Denoiser {
    NDArray forward(NDArray x, NDArray cond, NDArray t);
}
