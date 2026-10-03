//! s08/s09: probabilities and decisions with the reference's precisions (laya/agent.py:772-813,
//! laya/common.py:338-364), as observed in the oracle environment (NumPy 2.4.6, NEP 50):
//! - `logits[r,:k] / t_scale`: the Python float is cast to float32, math in float32;
//! - `np.exp` / `np.log` / sums in float32; sums use NumPy's pairwise algorithm;
//! - `ent / math.log(k)` is float32 (observed V-24); the score expectation is float64;
//! - noul confidence `max(p1, 1 - p1)` is float64.
//! `np.exp` / `np.log` are NumPy's own float32 SIMD kernels, ported bit-exactly in `npmath`
//! (libm differs by 1-2 ULP; that was deviation D-05, measured on v001 before this port).

use crate::npmath;

/// NumPy `pairwise_sum` for float32 (numpy/_core/src/umath/loops_utils.h.src), block size 128.
pub fn pairwise_sum_f32(a: &[f32]) -> f32 {
    let n = a.len();
    if n < 8 {
        let mut res = -0.0f32;
        for &x in a {
            res += x;
        }
        res
    } else if n <= 128 {
        let mut r = [0f32; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum_f32(&a[..n2]) + pairwise_sum_f32(&a[n2..])
    }
}

/// `pairwise_sum` for float64 (same algorithm, used by the score expectation).
pub fn pairwise_sum_f64(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = -0.0f64;
        for &x in a {
            res += x;
        }
        res
    } else if n <= 128 {
        let mut r = [0f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum_f64(&a[..n2]) + pairwise_sum_f64(&a[n2..])
    }
}

/// `z = logits[:k] / t; p = exp(z - max(z)); p /= sum(p)` in float32.
pub fn tempered_softmax(logits: &[f32], t: f64) -> (Vec<f32>, Vec<f32>) {
    let t32 = t as f32;
    let z: Vec<f32> = logits.iter().map(|&l| l / t32).collect();
    let m = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = z.iter().map(|&v| npmath::expf(v - m)).collect();
    let s = pairwise_sum_f32(&e);
    let p = e.iter().map(|&v| v / s).collect();
    (z, p)
}

/// `int(p.argmax())`: the first maximum (Rust's `max_by` would return the last).
pub fn argmax_first(p: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in p.iter().enumerate() {
        if v > p[best] {
            best = i;
        }
    }
    best
}

/// Lab-defined ranking: descending p, ties by ascending index (docs/parity-definition.md §3).
pub fn ranking(p: &[f32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..p.len()).collect();
    idx.sort_by(|&a, &b| (p[b] as f64).partial_cmp(&(p[a] as f64)).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
    idx
}

/// `answer_confidence(p, k)`: `float(np.clip(np.max(p[:k]), 0, 1))`, 1.0 if k < 1.
pub fn answer_confidence(p: &[f32]) -> f64 {
    if p.is_empty() {
        return 1.0;
    }
    let m = p.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    m.clamp(0.0, 1.0) as f64
}

/// `confidence_from_probs(p, k)`: `clip(1 - H(p)/ln k, 0, 1)` computed in float32; 1.0 if k < 2.
pub fn confidence_from_probs(p: &[f32]) -> f64 {
    let k = p.len();
    if k < 2 {
        return 1.0;
    }
    let lo = 1e-12f64 as f32;
    let prod: Vec<f32> = p.iter().map(|&x| x * npmath::logf(x.clamp(lo, 1.0))).collect();
    let ent = -pairwise_sum_f32(&prod);
    let lnk = (k as f64).ln() as f32;
    let c = 1.0f32 - ent / lnk;
    c.clamp(0.0, 1.0) as f64
}

/// `float((np.arange(k) * p).sum())`: int64 * float32 promotes to float64.
pub fn expected_score(p: &[f32]) -> f64 {
    let terms: Vec<f64> = p.iter().enumerate().map(|(i, &x)| i as f64 * x as f64).collect();
    pairwise_sum_f64(&terms)
}

/// `torch.softmax(act.float(), -1)` on one row: max-subtract, exp, sum, multiply by the reciprocal.
pub fn act_softmax(row: &[f32]) -> Vec<f32> {
    let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = row.iter().map(|&v| (v - m).exp()).collect();
    let inv = 1.0f32 / e.iter().sum::<f32>();
    e.iter().map(|&v| v * inv).collect()
}

/// CPython `round(x, 4)`: correctly rounded on the exact binary value, ties to even, then parsed back.
pub fn round4(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.4}").parse::<f64>().expect("formatted float parses")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round4_matches_cpython_on_ties() {
        // CPython: round(0.15625, 4) == 0.1562, round(0.03125, 4) == 0.0312, round(0.00015, 4) == 0.0001 (binary < tie)
        assert_eq!(round4(0.15625), 0.1562);
        assert_eq!(round4(0.03125), 0.0312);
        assert_eq!(round4(0.00015), 0.0001);
        assert_eq!(round4(0.99995), 1.0);
        assert_eq!(round4(-0.00001), -0.0);
    }

    #[test]
    fn argmax_takes_the_first_tie() {
        assert_eq!(argmax_first(&[0.2, 0.4, 0.4]), 1);
        assert_eq!(ranking(&[0.2, 0.4, 0.4]), vec![1, 2, 0]);
    }

    #[test]
    fn pairwise_matches_sequential_below_eight() {
        let a = [0.1f32, 0.2, 0.3];
        assert_eq!(pairwise_sum_f32(&a), -0.0f32 + 0.1 + 0.2 + 0.3);
    }
}
