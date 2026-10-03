//! s03/s04: `build_sequence` (laya/common.py:94-146) and `collate_items` (laya/common.py:395-437).

use crate::normalize::QType;

/// One question row: `[CLS] head [SEP] ([MASK] opt)* [SEP] state [SEP]`, cut to `max_len`.
pub struct Row {
    pub ids: Vec<i64>,
    pub markers: Vec<i64>,
    /// Length of the prefix through the second `[SEP]` as the harness measures it
    /// (`len(build_sequence(..., state_ids=[])) - 1`).
    pub prefix_len: usize,
    pub qtype: QType,
}

pub struct Specials {
    pub cls: i64,
    pub sep: i64,
    pub mask: i64,
}

/// Exact port of the budget arithmetic. `opt_tokens` are the per-option ids from
/// `tok(" " + opt, truncation=True, max_length=48)` (= full ids `[:48]`), without the `[MASK]`.
pub fn build_sequence(
    sp: &Specials,
    head_ids_full: &[i64],
    opt_tokens: &[Vec<i64>],
    state_ids: &[i64],
    max_len: usize,
    head_max_len: usize,
    truncate_left: bool,
) -> (Vec<i64>, Vec<i64>) {
    let mut opt_ids: Vec<Vec<i64>> = opt_tokens
        .iter()
        .map(|o| {
            let mut v = vec![sp.mask];
            v.extend_from_slice(&o[..o.len().min(48)]);
            v
        })
        .collect();
    let total = |o: &Vec<Vec<i64>>| o.iter().map(Vec::len).sum::<usize>() as i64;
    let mut opt_budget = head_max_len as i64 - total(&opt_ids);
    if opt_budget < 16 {
        let n = opt_ids.len().max(1) as i64;
        let per = 4_i64.max((head_max_len as i64 - 16).div_euclid(n)) as usize;
        for o in &mut opt_ids {
            o.truncate(per);
        }
        opt_budget = head_max_len as i64 - total(&opt_ids);
    }
    let keep_head = 8_i64.max(opt_budget) as usize;
    let head = &head_ids_full[..head_ids_full.len().min(keep_head)];

    let mut ids = vec![sp.cls];
    ids.extend_from_slice(head);
    ids.push(sp.sep);
    let mut markers = Vec::with_capacity(opt_ids.len());
    for o in &opt_ids {
        markers.push(ids.len() as i64);
        ids.extend_from_slice(o);
    }
    ids.push(sp.sep);
    let room = (max_len as i64 - ids.len() as i64 - 1).max(0) as usize;
    // not state_ids[-room:]: with room 0 that would be the whole state (laya/common.py:143)
    let st: &[i64] = if truncate_left {
        &state_ids[state_ids.len().saturating_sub(room)..]
    } else {
        &state_ids[..state_ids.len().min(room)]
    };
    ids.extend_from_slice(st);
    ids.push(sp.sep);
    ids.truncate(max_len);
    markers.retain(|&m| (m as usize) < max_len);
    (ids, markers)
}

/// Collated batch, row-major, right-padded to the longest row (dynamic L).
pub struct Batch {
    pub n: usize,
    pub l: usize,
    pub kmax: usize,
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    pub marker_pos: Vec<i64>,
    pub marker_mask: Vec<bool>,
    pub qtype: Vec<i64>,
}

pub fn collate(rows: &[Row], pad: i64) -> Batch {
    let n = rows.len();
    let l = rows.iter().map(|r| r.ids.len()).max().unwrap_or(0);
    let kmax = rows.iter().map(|r| r.markers.len()).max().unwrap_or(0);
    let mut b = Batch {
        n,
        l,
        kmax,
        input_ids: vec![pad; n * l],
        attention_mask: vec![0; n * l],
        marker_pos: vec![0; n * kmax],
        marker_mask: vec![false; n * kmax],
        qtype: rows.iter().map(|r| r.qtype.code() as i64).collect(),
    };
    for (i, r) in rows.iter().enumerate() {
        b.input_ids[i * l..i * l + r.ids.len()].copy_from_slice(&r.ids);
        b.attention_mask[i * l..i * l + r.ids.len()].fill(1);
        b.marker_pos[i * kmax..i * kmax + r.markers.len()].copy_from_slice(&r.markers);
        b.marker_mask[i * kmax..i * kmax + r.markers.len()].fill(true);
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    const SP: Specials = Specials { cls: 1, sep: 2, mask: 3 };

    #[test]
    fn room_zero_keeps_no_state_even_when_truncating_left() {
        let (ids, markers) = build_sequence(&SP, &[10, 11], &[vec![20], vec![21]], &[30, 31, 32], 9, 256, true);
        // [CLS] 10 11 [SEP] [MASK] 20 [MASK] 21 [SEP] -> 9 ids fill max_len; room = 0; trailing [SEP] cut
        assert_eq!(ids, vec![1, 10, 11, 2, 3, 20, 3, 21, 2]);
        assert_eq!(markers, vec![4, 6]);
    }

    #[test]
    fn shrink_path_uses_per_including_mask() {
        let opts: Vec<Vec<i64>> = (0..70).map(|i| vec![100 + i, 200 + i, 300 + i, 400 + i, 500 + i]).collect();
        let (ids, markers) = build_sequence(&SP, &[10; 20], &opts, &[], 1024, 256, false);
        // per = max(4, 240 // 70) = 4 -> each option is [MASK] + 3 tokens; head floored at 8
        assert_eq!(markers.len(), 70);
        assert_eq!(markers[1] - markers[0], 4);
        assert_eq!(ids.len(), 1 + 8 + 1 + 70 * 4 + 1 + 1);
    }
}
