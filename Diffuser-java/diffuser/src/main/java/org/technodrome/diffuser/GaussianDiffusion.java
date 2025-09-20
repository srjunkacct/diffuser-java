package org.technodrome.diffuser;

public class GaussianDiffusion {

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

    public GaussianDiffusion(int horizon,
                             int observationDimension,
                             int actionDimension,
                             int timesteps,
                             String lossType,
                             boolean clipDenoised,
                             boolean predictEpsilon,
                             double actionWeight,
                             double lossDiscount) {
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
    }
}
