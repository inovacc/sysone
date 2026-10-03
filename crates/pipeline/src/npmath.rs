//! Bit-exact ports of NumPy's float32 `exp` and `log` ufunc kernels, so the Rust decode reproduces the
//! reference's probabilities to the bit (resolves deviation D-05).
//!
//! Source: numpy v2.4.6 `numpy/_core/src/umath/loops_exponent_log.dispatch.c.src` (`simd_exp_FLOAT`,
//! `simd_log_FLOAT`, the `SIMD_AVX2_FMA3` helpers) with constants from `npy_simd_data.h` and `npy_math.h`.
//! NumPy dispatches this kernel on x86-64-v3 (AVX2 + FMA3), the oracle host's CPU class. SIMD lanes are
//! independent, so one scalar lane with `f32::mul_add` for every `_mm256_fmadd_ps` gives the same bits.
//! CPUs that dispatch the AVX-512F kernel use the same algorithm but a different `scalef`; V-10/V-21 track
//! cross-class parity.

const RINT_CVT_MAGIC: f32 = 12582912.0; // 0x1.800000p+23f
const CODY_WAITE_LOGE_2_HIGH: f32 = -6.93145752e-1;
const CODY_WAITE_LOGE_2_LOW: f32 = -1.42860677e-6;
const EXP_P0: f32 = 9.999999999980870924916e-01;
const EXP_P1: f32 = 7.257664613233124478488e-01;
const EXP_P2: f32 = 2.473615434895520810817e-01;
const EXP_P3: f32 = 5.114512081637298353406e-02;
const EXP_P4: f32 = 6.757896990527504603057e-03;
const EXP_P5: f32 = 5.082762527590693718096e-04;
const EXP_Q0: f32 = 1.000000000000000000000e+00;
const EXP_Q1: f32 = -2.742335390411667452936e-01;
const EXP_Q2: f32 = 2.159509375685829852307e-02;
const LOG2E: f32 = 1.442695040888963407359924681001892137;
const XMAX: f32 = 88.72283935546875;
const XMIN: f32 = -103.97208404541015625;

const LOG_P0: f32 = 0.000000000000000000000e+00;
const LOG_P1: f32 = 9.999999999999998702752e-01;
const LOG_P2: f32 = 2.112677543073053063722e+00;
const LOG_P3: f32 = 1.480000633576506585156e+00;
const LOG_P4: f32 = 3.808837741388407920751e-01;
const LOG_P5: f32 = 2.589979117907922693523e-02;
const LOG_Q0: f32 = 1.000000000000000000000e+00;
const LOG_Q1: f32 = 2.612677543073109236779e+00;
const LOG_Q2: f32 = 2.453006071784736363091e+00;
const LOG_Q3: f32 = 9.864942958519418960339e-01;
const LOG_Q4: f32 = 1.546476374983906719538e-01;
const LOG_Q5: f32 = 5.875095403124574342950e-03;
const LOGE2: f32 = 0.693147180559945309417232121458176568;
const SQRT1_2: f32 = 0.707106781186547524400844362104849039;

/// `_mm256_cvtps_epi32` under the default MXCSR rounding (round to nearest even).
fn cvtps_epi32(x: f32) -> i32 {
    x.round_ties_even() as i32
}

/// `fma_scalef_ps`: poly * 2^quadrant by adding to the exponent bits, with the denormal split.
fn scalef(poly: f32, quadrant: f32) -> f32 {
    let min_q = -125.0f32;
    if quadrant <= min_q {
        let quad_diff = 0.0f32 - (quadrant - min_q);
        let two_power_diff = 1i32.wrapping_shl(cvtps_epi32(quad_diff) as u32);
        let q = quadrant.max(min_q);
        let exponent = cvtps_epi32(q).wrapping_shl(23);
        let p = f32::from_bits((poly.to_bits() as i32).wrapping_add(exponent) as u32);
        p / two_power_diff as f32
    } else {
        let exponent = cvtps_epi32(quadrant).wrapping_shl(23);
        f32::from_bits((poly.to_bits() as i32).wrapping_add(exponent) as u32)
    }
}

/// NumPy float32 `np.exp` (simd_exp_FLOAT, one lane).
pub fn expf(x_in: f32) -> f32 {
    if x_in.is_nan() {
        return f32::NAN;
    }
    if x_in >= XMAX {
        return f32::INFINITY;
    }
    if x_in <= XMIN {
        return 0.0;
    }
    let mut quadrant = x_in * LOG2E;
    quadrant = quadrant + RINT_CVT_MAGIC;
    quadrant = quadrant - RINT_CVT_MAGIC;
    // Cody-Waite range reduction: fmadd(y, c1, x); fmadd(y, c2, .); fmadd(y, 0, .)
    let mut x = quadrant.mul_add(CODY_WAITE_LOGE_2_HIGH, x_in);
    x = quadrant.mul_add(CODY_WAITE_LOGE_2_LOW, x);
    x = quadrant.mul_add(0.0, x);
    let mut num = EXP_P5.mul_add(x, EXP_P4);
    num = num.mul_add(x, EXP_P3);
    num = num.mul_add(x, EXP_P2);
    num = num.mul_add(x, EXP_P1);
    num = num.mul_add(x, EXP_P0);
    let mut den = EXP_Q2.mul_add(x, EXP_Q1);
    den = den.mul_add(x, EXP_Q0);
    scalef(num / den, quadrant)
}

/// `fma_get_exponent` (with the denormal pre-scale).
fn get_exponent(x: f32) -> f32 {
    let denormal = x < f32::MIN_POSITIVE;
    let v = if denormal { x * f32::from_bits(0x7180_0000) } else { x };
    let e = ((v.to_bits() >> 23) as i32 - 0x7E) as f32;
    if denormal { e - 100.0 } else { e }
}

/// `fma_get_mantissa`: mantissa bits with exponent 126 (value in [0.5, 1)).
fn get_mantissa(x: f32) -> f32 {
    let v = if x < f32::MIN_POSITIVE { x * f32::from_bits(0x7180_0000) } else { x };
    f32::from_bits((v.to_bits() & 0x7f_ffff) | (126 << 23))
}

/// NumPy float32 `np.log` (simd_log_FLOAT, one lane).
pub fn logf(x_in: f32) -> f32 {
    if x_in.is_nan() {
        return f32::NAN;
    }
    if x_in < 0.0 {
        return -f32::NAN;
    }
    if x_in == 0.0 {
        return f32::NEG_INFINITY;
    }
    if x_in == f32::INFINITY {
        return f32::INFINITY;
    }
    let mut exponent = get_exponent(x_in);
    let mut x = get_mantissa(x_in);
    if x <= SQRT1_2 {
        x = x + x;
        exponent = exponent - 1.0;
    }
    x = x - 1.0;
    let mut num = LOG_P5.mul_add(x, LOG_P4);
    num = num.mul_add(x, LOG_P3);
    num = num.mul_add(x, LOG_P2);
    num = num.mul_add(x, LOG_P1);
    num = num.mul_add(x, LOG_P0);
    let mut den = LOG_Q5.mul_add(x, LOG_Q4);
    den = den.mul_add(x, LOG_Q3);
    den = den.mul_add(x, LOG_Q2);
    den = den.mul_add(x, LOG_Q1);
    den = den.mul_add(x, LOG_Q0);
    exponent.mul_add(LOGE2, num / den)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_and_log_are_close_to_libm() {
        // Sanity only (exactness vs NumPy is checked by the numpy corpus, .scripts/31-B_*):
        for &x in &[-20.0f32, -3.5, -1.0, -0.25, 0.0, 0.5, 1.0, 10.0] {
            let rel = ((expf(x) - x.exp()) / x.exp()).abs();
            assert!(rel < 5e-7, "exp {x}: {} vs {}", expf(x), x.exp());
        }
        for &x in &[1e-12f32, 1e-3, 0.3, 0.5, 0.9, 1.0, 2.0, 100.0] {
            assert!((logf(x) - x.ln()).abs() < 5e-6, "log {x}: {} vs {}", logf(x), x.ln());
        }
        assert_eq!(expf(0.0), 1.0);
        assert_eq!(logf(1.0), 0.0);
    }
}
