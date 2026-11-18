/* Receiver hot path in C: FIR matched filter, Gardner timing loop, QAM slicer.
 *
 * The timing loop is the reason this file exists. Each strobe depends on the
 * loop state left by the previous one, so there is no array operation that
 * computes them together - NumPy has nothing to offer a recurrence, and the
 * Python version spends its time in the interpreter rather than on arithmetic.
 * The filter and the slicer are here for comparison, and the benchmark reports
 * honestly that for those two the gain over NumPy is small, because NumPy was
 * already calling compiled code.
 *
 * Layout matches numpy complex128 exactly (two contiguous float64, real
 * first), so arrays cross the boundary by pointer with no copy or conversion.
 *
 * Build: cc -O2 -shared -fPIC -o libreceiver.so receiver.c -lm
 */
#include <complex.h>
#include <math.h>
#include <stdint.h>

typedef double _Complex cplx;

/* Full convolution of a complex signal with real taps.
 * y must hold nx + nh - 1 elements. */
void mf_filter(const cplx *x, int64_t nx, const double *h, int64_t nh, cplx *y)
{
    const int64_t ny = nx + nh - 1;
    for (int64_t n = 0; n < ny; ++n) {
        /* Clamp the tap range instead of testing inside the inner loop. */
        int64_t k0 = n - nx + 1;
        if (k0 < 0) k0 = 0;
        int64_t k1 = n < nh - 1 ? n : nh - 1;
        cplx acc = 0.0;
        for (int64_t k = k0; k <= k1; ++k)
            acc += h[k] * x[n - k];
        y[n] = acc;
    }
}

/* Cubic Lagrange interpolation at continuous index t. Returns 0 at the edges,
 * matching the Python reference exactly so the two are bit-comparable. */
static inline cplx interp(const cplx *x, int64_t nx, double t)
{
    const double fl = floor(t);
    const int64_t n = (int64_t)fl;
    const double mu = t - fl;
    if (n - 1 < 0 || n + 2 >= nx) return 0.0;

    const double a = -mu * (mu - 1.0) * (mu - 2.0) / 6.0;
    const double b = (mu + 1.0) * (mu - 1.0) * (mu - 2.0) / 2.0;
    const double c = -(mu + 1.0) * mu * (mu - 2.0) / 2.0;
    const double d = (mu + 1.0) * mu * (mu - 1.0) / 6.0;
    return a * x[n - 1] + b * x[n] + c * x[n + 1] + d * x[n + 2];
}

/* Gardner timing-error detector driving a PI-controlled interpolator.
 *
 * e[k] = Re{ (y[k] - y[k-1]) * conj(y[k - sps/2]) }
 *
 * Decision-free and carrier-phase independent, so it locks before the carrier
 * loop does. Returns the number of symbols written. */
int64_t gardner(const cplx *x, int64_t nx, int sps, double kp, double ki,
                cplx *out, double *tau_out, double *err_out, int64_t n_sym)
{
    if (sps < 2 || n_sym < 1) return 0;

    double tau = 0.0, integ = 0.0;
    cplx prev = 0.0;
    const double limit = (double)sps;
    const double half = 0.5 * (double)sps;

    for (int64_t k = 0; k < n_sym; ++k) {
        const double base = (double)((k + 1) * sps) + tau;
        const cplx y = interp(x, nx, base);
        const cplx mid = interp(x, nx, base - half);
        const double e = creal((y - prev) * conj(mid));

        integ += e;
        tau -= kp * e + ki * integ;
        /* Clamp: an unbounded accumulator is how a timing loop ends up locked
         * one symbol off, showing a clean eye at BER 0.5. */
        if (tau > limit) tau = limit;
        else if (tau < -limit) tau = -limit;

        out[k] = y;
        tau_out[k] = tau;
        err_out[k] = e;
        prev = y;
    }
    return n_sym;
}

/* Square-QAM hard slicer. Per-dimension nearest-level decision, which is the
 * maximum-likelihood rule for a square constellation in AWGN. Bits are written
 * most significant first, I then Q. */
void demap_qam(const cplx *s, int64_t ns, double d, int n_levels,
               int bits_per_dim, const int64_t *level_to_label, uint8_t *bits)
{
    const double inv2d = 0.5 / d;
    const double bias = 0.5 * (double)(n_levels - 1);
    const int64_t top = n_levels - 1;
    const int64_t stride = 2 * bits_per_dim;

    for (int64_t i = 0; i < ns; ++i) {
        const double re = creal(s[i]), im = cimag(s[i]);
        /* nearbyint, not llround. The two differ on exact ties: llround rounds
         * half away from zero, while numpy's rint - and IEEE's default mode,
         * which nearbyint honours - rounds half to even. A sample landing
         * exactly on a decision boundary would then decode differently in C
         * and in Python, which destroys the bit-exactness the fast path is
         * only trustworthy because of. Ties have measure zero once there is
         * noise, which is precisely why this would never have shown up in a
         * BER number and had to be caught by a direct comparison. */
        int64_t qi = (int64_t)nearbyint(re * inv2d + bias);
        int64_t qq = (int64_t)nearbyint(im * inv2d + bias);
        if (qi < 0) qi = 0; else if (qi > top) qi = top;
        if (qq < 0) qq = 0; else if (qq > top) qq = top;

        const int64_t gi = level_to_label[qi], gq = level_to_label[qq];
        uint8_t *o = bits + i * stride;
        for (int b = 0; b < bits_per_dim; ++b) {
            const int sh = bits_per_dim - 1 - b;
            o[b] = (uint8_t)((gi >> sh) & 1);
            o[bits_per_dim + b] = (uint8_t)((gq >> sh) & 1);
        }
    }
}

/* Hamming distance between two unpacked bit arrays. Kept as a deliberate
 * negative result: the benchmark shows it is several times SLOWER than
 * numpy's vectorised comparison, because a single ctypes call costs more
 * than the whole operation. It is here to be measured, not to be used. */
int64_t bit_errors(const uint8_t *a, const uint8_t *b, int64_t n)
{
    int64_t c = 0;
    for (int64_t i = 0; i < n; ++i) c += (a[i] != b[i]);
    return c;
}
