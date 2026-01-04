package org.technodrome.diffuser.diffusion;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import org.jetbrains.annotations.Nullable;
import org.technodrome.diffuser.Denoiser;
import org.technodrome.diffuser.diffusion.helpers.NDUtils;

import java.util.List;
import java.util.Map;

import static org.technodrome.diffuser.diffusion.helpers.NDUtils.applyConditioning;
import static org.technodrome.diffuser.diffusion.helpers.NDUtils.extract;

public class GaussianDiffusion {

    private static NDManager manager = NDManager.newBaseManager(Device.gpu());
    private final Denoiser denoiserModel;
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
    private NDArray inverseSqrtAlphasCumprod;
    private NDArray inverseSqrtOneMinusAlphasCumprod;
    private NDArray posteriorMeanCoefficient1;
    private NDArray posteriorMeanCoefficient2;
    private NDArray posteriorVariance;
    private NDArray posteriorLogVarianceClipped;
    private NDArray sqrtAlphasCumProd;
    private NDArray sqrtOneMinusAlphasCumProd;

    public GaussianDiffusion(Denoiser denoiserModel,
                             int horizon,
                             int observationDimension,
                             int actionDimension,
                             int timesteps,
                             String lossType,
                             boolean clipDenoised,
                             boolean predictEpsilon,
                             double actionWeight,
                             double lossDiscount,
                             double[] lossWeightsByDimension) {
        this.denoiserModel = denoiserModel;
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
        NDArray betas = NDUtils.cosineBetaSchedule(this.timesteps);
        NDArray alphas = null;
        betas.copyTo(alphas);
        alphas = alphas.mul(-1.0).add(1.0);
        NDArray alphasCumProd = alphas.cumProd(0);

        NDIndex prevIndex = new NDIndex(":-1");
        NDArray alphasCumProdShift = alphasCumProd.get(prevIndex);
        NDArray alphasCumProdPrev = manager.ones(new Shape(List.of(1L)));
        alphasCumProdPrev.concat(alphasCumProdShift, 0);

        this.sqrtAlphasCumProd = alphasCumProd.sqrt();
        NDArray oneMinusAlphasCumProd = alphasCumProd.mul(-1).add(1.0);
        NDArray oneMinusAlphasCumProdPrev = alphasCumProdPrev.mul(-1).add(1.0);
        this.sqrtOneMinusAlphasCumProd = oneMinusAlphasCumProd.sqrt();
        NDArray logOneMinusAlphasCumProd = oneMinusAlphasCumProd.log();
        this.inverseSqrtAlphasCumprod = sqrtAlphasCumProd.inverse();
        this.inverseSqrtOneMinusAlphasCumprod = alphasCumProd.sub(1.0).inverse().sqrt();
        this.posteriorVariance = betas.mul(oneMinusAlphasCumProdPrev).div(oneMinusAlphasCumProd);
        this.posteriorLogVarianceClipped = posteriorVariance.log().clip(1e-20, Double.MAX_VALUE);
        this.posteriorMeanCoefficient1 = betas.mul(alphasCumProdPrev.sqrt()).div(oneMinusAlphasCumProd);
        this.posteriorMeanCoefficient2 = oneMinusAlphasCumProdPrev.mul(alphas).div(oneMinusAlphasCumProd);
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

    private NDArray predictStartFromNoise(NDArray xT, NDArray t, NDArray noise) {
        if (this.predictEpsilon) {
            return extract(this.inverseSqrtAlphasCumprod, t, xT.getShape()).mul(xT).sub(
                    extract(this.inverseSqrtOneMinusAlphasCumprod, t, xT.getShape()).mul(noise));
        }

        return noise;
    }

    private NDArray[] qPosterior(NDArray xStart, NDArray xT, NDArray t) {
        NDArray posteriorMean = extract(this.posteriorMeanCoefficient1, t, xT.getShape()).mul(xStart).add(
                extract(this.posteriorMeanCoefficient2, t, xT.getShape()).mul(xT));
        NDArray selectedPosteriorVariance = extract(this.posteriorVariance, t, xT.getShape());
        NDArray selectedLogPosteriorVarianceClipped = extract(this.posteriorLogVarianceClipped, t, xT.getShape());
        return new NDArray[]{posteriorMean, selectedPosteriorVariance, selectedLogPosteriorVarianceClipped};

    }

    private NDArray[] pMeanVariance(NDArray x, NDArray cond, NDArray t) {
        NDArray xRecon = predictStartFromNoise(x, t, this.denoiserModel.forward(x, cond, t, true));
        if (this.clipDenoised) {
            xRecon.clip(-1., -1.);
        }
        return this.qPosterior(xRecon, x, t);
    }

//
//    @torch.no_grad()
//    def p_sample_loop(self, shape, cond, verbose=True, return_chain=False, sample_fn=default_sample_fn, **sample_kwargs):
//    device = self.betas.device
//
//            batch_size = shape[0]
//    x = torch.randn(shape, device=device)
//    x = apply_conditioning(x, cond, self.action_dim)
//
//    chain = [x] if return_chain else None
//
//            progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
//            for i in reversed(range(0, self.n_timesteps)):
//    t = make_timesteps(batch_size, i, device)
//    x, values = sample_fn(self, x, cond, t, **sample_kwargs)
//    x = apply_conditioning(x, cond, self.action_dim)
//
//            progress.update({'t': i, 'vmin': values.min().item(), 'vmax': values.max().item()})
//            if return_chain: chain.append(x)
//
//            progress.stamp()
//
//    x, values = sort_by_values(x, values)
//        if return_chain: chain = torch.stack(chain, dim=1)
//            return Sample(x, values, chain)
//
//    @torch.no_grad()
//    def conditional_sample(self, cond, horizon=None, **sample_kwargs):
//            '''
//    conditions : [ (time, state), ... ]
//            '''
//    device = self.betas.device
//            batch_size = len(cond[0])
//    horizon = horizon or self.horizon
//    shape = (batch_size, horizon, self.transition_dim)
//
//            return self.p_sample_loop(shape, cond, **sample_kwargs)
//
//            #------------------------------------------ training ------------------------------------------#

    private NDArray qSample(NDArray xStart, Map<Integer, NDArray> t, @Nullable NDArray noise) {
        if (noise == null) {
            noise = manager.randomNormal(xStart.getShape());
        }

        return extract(this.sqrtAlphasCumProd, t, xStart.getShape()).mul(xStart).add(
                extract(this.sqrtOneMinusAlphasCumProd, t, xStart.getShape()).mul(noise));
    }


    private NDArray pLosses(NDArray xStart, Map<Integer, NDArray> cond, NDArray t) {

        NDArray noise = manager.randomNormal(xStart.getShape());
        NDArray xNoisy = this.qSample(xStart, cond, t);
        xNoisy = applyConditioning(xNoisy, cond, this.actionDimension);

        NDArray xRecon = this.denoiserModel.forward(xStart, t, noise, true);
        xRecon = applyConditioning(xRecon, cond, this.actionDimension);

        if (this.predictEpsilon) {
            return null;
        } else {
            return null;
        }
    }
//
//    def p_losses(self, x_start, cond, t):
//    noise = torch.randn_like(x_start)
//
//    x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
//    x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)
//
//    x_recon = self.model(x_noisy, cond, t)
//    x_recon = apply_conditioning(x_recon, cond, self.action_dim)
//
//        assert noise.shape == x_recon.shape
//
//        if self.predict_epsilon:
//    loss, info = self.loss_fn(x_recon, noise)
//            else:
//    loss, info = self.loss_fn(x_recon, x_start)
//
//            return loss, info
//
//    def loss(self, x, *args):
//    batch_size = len(x)
//    t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
//        return self.p_losses(x, *args, t)
//
//    def forward(self, cond, *args, **kwargs):
//            return self.conditional_sample(cond, *args, **kwargs)
//
//
//    class ValueDiffusion(GaussianDiffusion):
//
//    def p_losses(self, x_start, cond, target, t):
//    noise = torch.randn_like(x_start)
//
//    x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
//    x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)
//
//    pred = self.model(x_noisy, cond, t)
//
//    loss, info = self.loss_fn(pred, target)
//            return loss, info
//
//    def forward(self, x, cond, t):
//            return self.model(x, cond, t)


    public record Sample(NDArray trajectories, NDArray values, NDArray chains) {

    }
}
