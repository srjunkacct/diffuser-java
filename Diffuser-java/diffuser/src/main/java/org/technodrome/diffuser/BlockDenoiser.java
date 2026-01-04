package org.technodrome.diffuser;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.nn.Block;
import ai.djl.training.ParameterStore;

public class BlockDenoiser implements Denoiser {
    private final Block block;
    private final ParameterStore ps;

    public BlockDenoiser(Block block, ParameterStore ps) {
        this.block = block;
        this.ps = ps;
    }

    @Override
    public NDArray forward(NDArray x, Object cond, NDArray t, boolean isTraining) {
        // Option A: you pre-apply conditioning outside, so cond is unused by the net:
        // inputs = [x, t]
        // Option B: you encode cond to an NDArray and pass it too.
        // For now, assume cond is already an NDArray:
        NDArray condArr = (NDArray) cond;

        NDList out = block.forward(ps, new NDList(x, condArr, t), isTraining);
        return out.singletonOrThrow();
    }
}