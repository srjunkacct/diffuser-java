package org.technodrome.diffuser;

import ai.djl.ndarray.NDArray;

/**
 * @param x    (B, H, transitionDim)
 * @param cond conditioning info (your own type)
 * @param t    (B,) int64 timesteps
 * @param training whether in training mode
 * @return prediction shaped like x (usually epsilon-hat or x0-hat)
 */
public interface Denoiser {
    NDArray forward(NDArray x, Object cond, NDArray t, bool isTraining);
}
