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
}
