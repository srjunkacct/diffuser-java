
import ai.djl.training.ParameterStore;

public final class SimpleMlpDenoiser implements DenoiseModel {

    private final Block mlp;              // maps (D + tDim) -> D
    private final ParameterStore ps;
    private final int transitionDim;
    private final int tEmbedDim;

    public SimpleMlpDenoiser(NDManager manager, int transitionDim, int tEmbedDim) {
        this.transitionDim = transitionDim;
        this.tEmbedDim = tEmbedDim;

        // A basic MLP: (D + tDim) -> hidden -> hidden -> D
        this.mlp = new SequentialBlock()
                .add(Linear.builder().setUnits(256).build())
                .add(Activation::swish)
                .add(Linear.builder().setUnits(256).build())
                .add(Activation::swish)
                .add(Linear.builder().setUnits(transitionDim).build());

        // ParameterStore created by trainer typically; placeholder here
        this.ps = new ParameterStore(manager, false);
    }

    @Override
    public NDArray forward(NDArray x, Object condUnused, NDArray t, boolean training) {
        // x: (B,H,D), t: (B,)
        NDManager m = x.getManager();
        Shape xs = x.getShape();
        long B = xs.get(0);
        long H = xs.get(1);
        long D = xs.get(2);

        // 1) timestep embedding: (B, tEmbedDim)
        NDArray tEmb = sinusoidalTimeEmbedding(t.toType(DataType.FLOAT32, false), tEmbedDim);

        // 2) expand to (B,H,tEmbedDim)
        tEmb = tEmb.reshape(B, 1, tEmbedDim).broadcast(new Shape(B, H, tEmbedDim));

        // 3) concat: (B,H,D+tEmbedDim)
        NDArray inp = NDArrays.concat(new NDList(x, tEmb), -1);

        // 4) flatten to (B*H, D+tEmbedDim)
        NDArray flat = inp.reshape(B * H, D + tEmbedDim);

        // 5) MLP
        NDArray outFlat = mlp.forward(ps, new NDList(flat), training).singletonOrThrow();

        // 6) reshape back (B,H,D)
        return outFlat.reshape(B, H, D);
    }

    /** Standard sinusoidal embedding used in diffusion models. */
    private static NDArray sinusoidalTimeEmbedding(NDArray tFloat, int dim) {
        // tFloat: (B,)
        NDManager m = tFloat.getManager();
        long B = tFloat.getShape().get(0);

        int half = dim / 2;
        NDArray freqs = m.arange(half).toType(DataType.FLOAT32, false);
        // exp(-log(10000) * i/(half-1))
        NDArray exponent = freqs.mul(-Math.log(10000.0) / Math.max(half - 1, 1));
        NDArray scales = exponent.exp();                // (half,)
        NDArray args = tFloat.reshape(B, 1).mul(scales.reshape(1, half)); // (B,half)
        NDArray emb = NDArrays.concat(new NDList(args.cos(), args.sin()), -1); // (B,2*half)

        if (dim % 2 == 1) {
            emb = NDArrays.concat(new NDList(emb, m.zeros(new Shape(B, 1))), -1);
        }
        return emb;
    }
}
