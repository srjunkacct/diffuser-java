package org.technodrome.diffuser;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;

import java.util.List;

public class GaussianDiffusion {

    private static NDManager manager = NDManager.newBaseManager(Device.gpu());
    private static Utilities utilities = new Utilities(manager);
    private int horizon;
    private int observationDimension;
    private int actionDimension;
    private int transitionDimension;
    private int timesteps;
    private String lossType;
    private boolean clipDenoised;
    private boolean predictEpsilon;
    private double actionWeight;
    private double lossDiscount;
    private double[] lossWeightsByDimension;
    private NDArray lossWeights;

    public GaussianDiffusion(int horizon,
                             int observationDimension,
                             int actionDimension,
                             int timesteps,
                             String lossType,
                             boolean clipDenoised,
                             boolean predictEpsilon,
                             double actionWeight,
                             double lossDiscount,
                             double[] lossWeightsByDimension) {
        this.horizon = horizon;
        this.observationDimension = observationDimension;
        this.actionDimension = actionDimension;
        this.transitionDimension = this.observationDimension + this.actionDimension;
        this.timesteps = timesteps;
        this.lossType = lossType;
        this.clipDenoised = clipDenoised;
        this.predictEpsilon = predictEpsilon;
        this.actionWeight = actionWeight;
        this.lossDiscount = lossDiscount;
        this.lossWeightsByDimension = lossWeightsByDimension;
        initializeBuffers();

    }

    private void initializeBuffers() {
        NDArray betas = utilities.cosineBetaSchedule(this.timesteps);
        NDArray alphas = null;
        betas.copyTo(alphas);
        alphas = alphas.mul(-1.0).add(1.0);
        NDArray alphasCumProd = alphas.cumProd(0);

        NDIndex prevIndex = new NDIndex(":-1");
        NDArray alphasCumProdShift = alphasCumProd.get(prevIndex);
        NDArray alphasCumProdPrev = manager.ones(new Shape(List.of(1L)));
        alphasCumProdPrev.concat(alphasCumProdShift, 0);

        NDArray sqrtAlphasCumProd = alphasCumProd.sqrt();
        NDArray oneMinusAlphasCumProd = alphasCumProd.mul(-1).add(1.0);
        NDArray oneMinusAlphasCumProdPrev = alphasCumProdPrev.mul(-1).add(1.0);
        NDArray sqrtOneMinueAlphasCumProd = oneMinusAlphasCumProd.sqrt();
        NDArray logOneMinusAlphasCumProd = oneMinusAlphasCumProd.log();
        NDArray inverseSqrtAlphasCumprod = sqrtAlphasCumProd.inverse();
        NDArray inverseSqrtOneMinusAlphasCumprod = alphasCumProd.sub(1.0).inverse().sqrt();
        NDArray posteriorVariance = betas.mul(oneMinusAlphasCumProdPrev).div(oneMinusAlphasCumProd);
        NDArray posteriorLogVarianceClamped = posteriorVariance.log().clip(1e-20, Double.MAX_VALUE);
        NDArray posteriorMeanCoefficient1 = betas.mul(alphasCumProdPrev.sqrt()).div(oneMinusAlphasCumProd);
        NDArray posteriorMeanCoefficient2 = oneMinusAlphasCumProdPrev.mul(alphas).div(oneMinusAlphasCumProd);
        this.lossWeights = getLossWeights(this.actionWeight, this.lossDiscount, this.lossWeightsByDimension);
    }

    private NDArray getLossWeights(double actionWeight,
                                   double lossDiscount,
                                   double[] lossWeightsByDimension) {
        this.actionWeight = actionWeight;
        NDArray dimensionWeights = manager.ones(new Shape(List.of((long) this.transitionDimension)));
        if (lossWeightsByDimension == null)
            lossWeightsByDimension = new double[]{};

        for (int index = 0; index < lossWeightsByDimension.length; index++) {
            dimensionWeights.get(this.actionDimension + index).mul(lossWeightsByDimension[index]);
        }

        NDArray discounts = manager.arange((float) this.horizon).mul(Math.log(lossDiscount)).exp();
        double discountsMean = discounts.mean().getDouble(0L);
        discounts = discounts.div(discountsMean);
        NDArray discounts2d = discounts.reshape(discounts.getShape().get(0), 1);
        NDArray dimWeights2d = manager.create(lossWeightsByDimension).reshape(1, lossWeightsByDimension.length);

        // Create the outer product of discounts and lossWeightsByDimension
        NDArray lossWeights = discounts2d.mul(dimWeights2d);
        // Set the first row equal to actionWeight
        lossWeights.set(new NDIndex("0, :{}", this.actionDimension), actionWeight);
        return lossWeights;
    }
}
