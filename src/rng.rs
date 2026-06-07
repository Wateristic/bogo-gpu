/// Per-index independent RNG.
///
/// Each shuffle index `i` derives its own seed via:
///   index_seed = seed.wrapping_add(i.wrapping_mul(0x9e3779b97f4a7c15))
/// That seed is then expanded with 2× splitmix64 → xoshiro128++.
/// A fresh [1..=25] array is initialized and shuffled exactly once.
/// No state carries across indices — every index is fully independent.
///
/// IMPORTANT: next_bounded uses rejection sampling, NOT Lemire's method.
/// The server verifies shuffles using this exact algorithm — any change
/// to the bounded RNG will produce verify_mismatch rejections.
 
pub struct Rng {
    state: [u32; 4],
}
 
impl Rng {
    pub fn new(seed: u64, index: u64) -> Self {
        const GOLDEN: u64 = 0x9e3779b97f4a7c15;
        let index_seed = seed.wrapping_add(index.wrapping_mul(GOLDEN));
        let z1 = splitmix64(index_seed.wrapping_add(GOLDEN));
        let z2 = splitmix64(index_seed.wrapping_add(GOLDEN.wrapping_mul(2)));
        Rng {
            state: [
                z1 as u32,
                (z1 >> 32) as u32,
                z2 as u32,
                (z2 >> 32) as u32,
            ],
        }
    }
 
    pub fn next_u32(&mut self) -> u32 {
        let result = self.state[0]
            .wrapping_add(self.state[3])
            .rotate_left(7)
            .wrapping_add(self.state[0]);
        let t = self.state[1] << 9;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(11);
        result
    }
 
    /// Rejection-sampling bounded RNG — must match the server's verifier exactly.
    pub fn next_bounded(&mut self, n: u32) -> u32 {
        let threshold = ((1u64 << 32) % n as u64) as u32;
        loop {
            let val = self.next_u32();
            if val >= threshold {
                return val % n;
            }
        }
    }
 
    pub fn shuffle(&mut self) -> [u8; 25] {
        let mut arr = init_arr();
        for i in (1..=24usize).rev() {
            let j = self.next_bounded(i as u32 + 1) as usize;
            arr.swap(i, j);
        }
        arr
    }
}
 
fn splitmix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}
 
pub fn init_arr() -> [u8; 25] {
    core::array::from_fn(|i| (i + 1) as u8)
}
 
pub fn count_fixed_points(arr: &[u8; 25]) -> u32 {
    arr.iter()
        .enumerate()
        .filter(|&(i, &v)| v == (i + 1) as u8)
        .count() as u32
}
 