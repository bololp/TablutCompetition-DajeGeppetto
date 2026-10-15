package it.unibo.ai.didattica.competition.tablut.training;

/**
 * Adam (Kingma &amp; Ba) su parametri {@code float[]}. Momenti in {@code float}; la correzione del bias usa {@code double}.
 * Separato dal trainer per poterlo confrontare con un'implementazione ingenua in {@code double} ({@code CnnTrainingCheck}).
 */
public final class Adam {

	private final float[] m, v;
	private long t;
	private final double beta1, beta2, eps;

	public Adam(int n) {
		this(n, 0.9, 0.999, 1e-8);
	}

	public Adam(int n, double beta1, double beta2, double eps) {
		this.m = new float[n];
		this.v = new float[n];
		this.beta1 = beta1;
		this.beta2 = beta2;
		this.eps = eps;
	}

	/** Un passo: {@code w -= lr * mhat / (sqrt(vhat) + eps)} con gradiente {@code g * gradScale} (es. 1/dimensioneBatch). */
	public void step(float[] w, float[] g, double lr, double gradScale) {
		t++;
		final double c1 = 1 - Math.pow(beta1, t), c2 = 1 - Math.pow(beta2, t);
		final float b1 = (float) beta1, b2 = (float) beta2, nb1 = 1 - b1, nb2 = 1 - b2, gs = (float) gradScale;
		final double a = lr / c1, sq = 1.0 / Math.sqrt(c2);
		for (int i = 0; i < w.length; i++) {
			final float gi = g[i] * gs;
			m[i] = b1 * m[i] + nb1 * gi;
			v[i] = b2 * v[i] + nb2 * gi * gi;
			w[i] -= (float) (a * m[i] / (Math.sqrt(v[i]) * sq + eps));
		}
	}
}
