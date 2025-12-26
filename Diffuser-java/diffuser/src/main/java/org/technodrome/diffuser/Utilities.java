package org.technodrome.diffuser;


import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import java.util.Map;

public class Utilities {

    private NDManager manager;

    public Utilities(NDManager manager) {
        this.manager = manager;
    }

    /**
     * Return a cosine beta schedule with the given number of steps
     *
     * @param steps the number of steps
     * @return the cosine beta schedule, as a double array of size steps
     */
    public NDArray cosineBetaSchedule(int steps) {
        double s = 0.008;
        int stepsPlusOne = steps + 1;
        NDArray alphas = manager.linspace(0, stepsPlusOne, stepsPlusOne);
        alphas = alphas.div(stepsPlusOne);
        alphas = alphas.add(s);
        alphas = alphas.mul(Math.PI * 0.5 / (1.0 + s));
        alphas = alphas.cos();
        alphas = alphas.pow(2);
        NDIndex futureIndex = new NDIndex("1:");
        NDIndex prevIndex = new NDIndex(":-1");
        NDArray betas = alphas.get(futureIndex);
        betas = betas.div(alphas.get(prevIndex));
        betas = betas.mul(-1.0);
        betas = betas.add(1.0);
        betas = betas.clip(0.0, 0.999);
        return betas;
    }

    /**
     * Extract elements from the tensor a according to the times in tensor t,
     * and reshape it as a tensor with rank equal to the rank of xShape.
     *
     * @param a
     * @param t      1 rank-1 tensor (vector) containing a list of times
     * @param xShape the shape the output should take
     * @return
     */
    public static NDArray extract(NDArray a, NDArray t, Shape xShape) {
        // t is usually shape (b,) and integer typed
        long b = t.getShape().get(0);

        // Make sure indices are INT64 (DJL likes int64 indices for take/gather)
        NDArray tIdx = t.toType(DataType.INT64, false);

        // Equivalent to a[t] in the common case (a is 1-D)
        NDArray out = a.take(tIdx); // out shape: (b,)

        // Reshape to (b, 1, 1, ..., 1) with same rank as xShape
        int ones = xShape.dimension() - 1;
        long[] newDims = new long[1 + ones];
        newDims[0] = b;
        for (int i = 1; i < newDims.length; i++) newDims[i] = 1;

        return out.reshape(new Shape(newDims));
    }

    public static NDArray applyConditioning(NDArray x,
                                            Map<Double, NDArray> conditions,
                                            int action_dim)
    {
        for ( Map.Entry<Double, NDArray> entry : conditions.entrySet() )
        {
            x[:,t, action_dim:] = entry.getValue().clone();
        }
        return x;
    }



//    class WeightedLoss(nn.Module):
//
//    def __init__(self, weights, action_dim):
//            super().__init__()
//        self.register_buffer('weights', weights)
//    self.action_dim = action_dim
//
//    def forward(self, pred, targ):
//            '''
//    pred, targ : tensor
//                [ batch_size x horizon x transition_dim ]
//            '''
//    loss = self._loss(pred, targ)
//    weighted_loss = (loss * self.weights).mean()
//    a0_loss = (loss[:, 0, :self.action_dim] / self.weights[0, :self.action_dim]).mean()
//        return weighted_loss, {'a0_loss': a0_loss}
//
//    class ValueLoss(nn.Module):
//    def __init__(self, *args):
//            super().__init__()
//
//    def forward(self, pred, targ):
//    loss = self._loss(pred, targ).mean()
//
//        if len(pred) > 1:
//    corr = np.corrcoef(
//            utils.to_np(pred).squeeze(),
//                utils.to_np(targ).squeeze()
//            )[0,1]
//                    else:
//    corr = np.NaN
//
//            info = {
//            'mean_pred': pred.mean(), 'mean_targ': targ.mean(),
//            'min_pred': pred.min(), 'min_targ': targ.min(),
//            'max_pred': pred.max(), 'max_targ': targ.max(),
//            'corr': corr,
//}
//
//        return loss, info
//
//class WeightedL1(WeightedLoss):
//
//        def _loss(self, pred, targ):
//        return torch.abs(pred - targ)
//
//class WeightedL2(WeightedLoss):
//
//        def _loss(self, pred, targ):
//        return F.mse_loss(pred, targ, reduction='none')
//
//class ValueL1(ValueLoss):
//
//        def _loss(self, pred, targ):
//        return torch.abs(pred - targ)
//
//class ValueL2(ValueLoss):
//
//        def _loss(self, pred, targ):
//        return F.mse_loss(pred, targ, reduction='none')




    public record Losses( WeightedL1 weightedL1
                          WeightedL2 weightedL2,
                          ValueL1 valueL1,
                          ValueL2 valueL2);
}
