// src/compute/gpu/mod.rs  —  CUDA backend (--features cuda)

use anyhow::{Context as _, Result};
use cust::{
    context::{Context, CurrentContext},
    device::Device,
    memory::{AsyncCopyDestination, DeviceBuffer, LockedBuffer},
    module::Module,
    prelude::*,
    stream::{Stream, StreamFlags},
    CudaFlags,
};

static PTX: &str = include_str!(env!("KERNEL_PTX_PATH"));

struct Slot {
    stream:               Stream,
    best_and_bid_dev:     DeviceBuffer<u64>,
    indices_dev:          DeviceBuffer<u64>,
    arrays_dev:           DeviceBuffer<u8>,
    global_threshold_dev: DeviceBuffer<u32>,
    best_and_bid_host:    LockedBuffer<u64>,
    indices_host:         LockedBuffer<u64>,
    arrays_host:          LockedBuffer<u8>,
    base_index: u64,
    in_flight:  bool,
}

impl Slot {
    fn new(blocks: usize) -> Result<Self> {
        Ok(Self {
            stream:               Stream::new(StreamFlags::NON_BLOCKING, None)?,
            best_and_bid_dev:     DeviceBuffer::zeroed(1)?,
            indices_dev:          DeviceBuffer::zeroed(blocks)?,
            arrays_dev:           DeviceBuffer::zeroed(blocks * 25)?,
            global_threshold_dev: DeviceBuffer::zeroed(1)?,
            best_and_bid_host:    LockedBuffer::new(&0u64, 1)?,
            indices_host:         LockedBuffer::new(&0u64, blocks)?,
            arrays_host:          LockedBuffer::new(&0u8,  blocks * 25)?,
            base_index: 0,
            in_flight:  false,
        })
    }

    fn launch(
        &mut self,
        func:          &cust::function::Function,
        seed:          u64,
        base:          u64,
        count:         u32,
        chunk:         u32,
        blocks:        u32,
        threads:       u32,
        min_threshold: u32,
    ) -> Result<()> {
        debug_assert!(!self.in_flight);
        self.base_index = base;
        self.best_and_bid_dev.set_8(0)?;
        // Reset global threshold to min_threshold for this launch.
        let min_thresh_host = [min_threshold];
        self.global_threshold_dev.copy_from(&min_thresh_host)?;
        let stream = &self.stream;
        unsafe {
            launch!(
                func<<<blocks, threads, 0, stream>>>(
                    seed, base, count, chunk,
                    self.best_and_bid_dev.as_device_ptr(),
                    self.indices_dev.as_device_ptr(),
                    self.arrays_dev.as_device_ptr(),
                    min_threshold,
                    self.global_threshold_dev.as_device_ptr()
                )
            )?;
            self.best_and_bid_dev.async_copy_to(&mut self.best_and_bid_host[..], stream)?;
            self.indices_dev     .async_copy_to(&mut self.indices_host     [..], stream)?;
            self.arrays_dev      .async_copy_to(&mut self.arrays_host      [..], stream)?;
        }
        self.in_flight = true;
        Ok(())
    }

    fn finish(&mut self, fallback_base: u64) -> Result<(u32, u64, [u8; 25])> {
        self.stream.synchronize()?;
        self.in_flight = false;
        let packed   = self.best_and_bid_host[0];
        let score    = (packed >> 32) as u32;
        let block_id = (packed & 0xffff_ffff) as usize;
        if score == 0 {
            return Ok((0, fallback_base, std::array::from_fn(|i| (i + 1) as u8)));
        }
        let best_index = self.indices_host[block_id];
        let best_arr   = std::array::from_fn::<u8, 25, _>(|p| {
            self.arrays_host[block_id * 25 + p]
        });
        Ok((score, best_index, best_arr))
    }

    fn ensure_idle(&mut self, fallback_base: u64) -> Result<Option<(u32, u64, [u8; 25])>> {
        if self.in_flight { Ok(Some(self.finish(fallback_base)?)) } else { Ok(None) }
    }
}

pub struct CudaWorker {
    _ctx:           Context,
    module:         Module,
    slots:          Vec<Slot>,
    pub blocks:     u32,
    pub threads:    u32,
    pub chunk_size: u32,
    // Moving average: ring buffer of recent best scores for min_threshold
    score_history:  [u32; 32],
    history_pos:    usize,
    history_count:  usize,
}

impl CudaWorker {
    pub fn new(blocks: u32, threads: u32, chunk_size: u32) -> Result<Self> {
        Self::new_with_slots(blocks, threads, chunk_size, 3)
    }

    pub fn new_with_slots(blocks: u32, threads: u32, chunk_size: u32, n_slots: usize) -> Result<Self> {
        eprintln!("[cuda] init");
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(0).context("no CUDA device found")?;
        let ctx    = Context::new(device)?;
        CurrentContext::set_current(&ctx)?;
        let module = Module::from_ptx(PTX, &[])?;
        let slots = (0..n_slots).map(|_| Slot::new(blocks as usize)).collect::<Result<Vec<_>>>()?;
        eprintln!("[cuda] ready — {}-buffered, {} blocks", n_slots, blocks);
        Ok(Self { _ctx: ctx, module, slots, blocks, threads, chunk_size,
                  score_history: [0u32; 32], history_pos: 0, history_count: 0 })
    }

    fn record_score(history: &mut [u32; 32], pos: &mut usize, count: &mut usize, score: u32) {
        if score == 0 { return; }
        history[*pos] = score;
        *pos = (*pos + 1) % 32;
        if *count < 32 { *count += 1; }
    }

    fn min_threshold(&self) -> u32 {
        if self.history_count < 8 { return 0; }
        let sum: u32 = self.score_history[..self.history_count].iter().sum();
        let avg = sum / self.history_count as u32;
        avg.saturating_sub(1)
    }

    pub fn run_batch(&mut self, seed: u64, base_index: u64, count: u32) -> Result<(u32, u64, [u8; 25])> {
        CurrentContext::set_current(&self._ctx)?;
        let func   = self.module.get_function("bogo_shuffle_kernel")?;
        let stride = (self.blocks as u64) * (self.threads as u64) * (self.chunk_size as u64);
        let total  = count as u64;
        if total == 0 {
            return Ok((0, base_index, std::array::from_fn(|i| (i + 1) as u8)));
        }
        let num_slots = self.slots.len();
        let mut best_c:    u32  = 0;
        let mut best_i:    u64  = base_index;
        let mut best_arr         = std::array::from_fn::<u8, 25, _>(|i| (i + 1) as u8);
        let mut next_slot: usize = 0;
        let mut offset:    u64   = 0;
        let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();

        // Borrow these fields directly so record_score doesn't need &mut self
        // while func holds an immutable borrow on self.module.
        let history  = &mut self.score_history;
        let hist_pos = &mut self.history_pos;
        let hist_cnt = &mut self.history_count;

        loop {
            if offset < total && queue.len() < num_slots {
                let slot_idx  = next_slot % num_slots;
                let sub_count = stride.min(total - offset) as u32;
                let sub_base  = base_index + offset;
                if let Some((bc, bi, ba)) = self.slots[slot_idx].ensure_idle(base_index)? {
                    if bc > best_c { best_c = bc; best_i = bi; best_arr = ba; }
                    Self::record_score(history, hist_pos, hist_cnt, bc);
                    queue.retain(|&s| s != slot_idx);
                }
                let min_thresh = if *hist_cnt < 8 { 0 } else {
                    let sum: u32 = history[..*hist_cnt].iter().sum();
                    (sum / *hist_cnt as u32).saturating_sub(1)
                };
                self.slots[slot_idx].launch(&func, seed, sub_base, sub_count,
                    self.chunk_size, self.blocks, self.threads, min_thresh)?;
                queue.push_back(slot_idx);
                next_slot += 1;
                offset    += sub_count as u64;
                continue;
            }
            if let Some(slot_idx) = queue.pop_front() {
                let (bc, bi, ba) = self.slots[slot_idx].finish(base_index)?;
                if bc > best_c { best_c = bc; best_i = bi; best_arr = ba; }
                Self::record_score(history, hist_pos, hist_cnt, bc);
            }
            if offset >= total && queue.is_empty() { break; }
        }
        Ok((best_c, best_i, best_arr))
    }
}
