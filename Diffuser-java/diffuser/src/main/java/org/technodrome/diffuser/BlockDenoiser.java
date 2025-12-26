import ai.djl.Model;
import ai.djl.ndarray.*;
import ai.djl.training.ParameterStore;
import ai.djl.nn.Block;

public class BlockDenoiser implements Denoiser {
    private final Block block;
    private final ParameterStore ps;

    public BlockDenoiseModel(Block block, ParameterStore ps) {
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

        NDList out = block.forward(ps, new NDList(x, condArr, t), training);
        return out.singletonOrThrow();
    }
}