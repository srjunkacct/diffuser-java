package org.technodrome.diffuser;


import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;

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
}
