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
    stream:            Stream,
    best_and_bid_dev:  DeviceBuffer<u64>,
    indices_dev:       DeviceBuffer<u64>,
    best_and_bid_host: LockedBuffer<u64>,
    indices_host:      LockedBuffer<u64>,
    base_index: u64,
    in_flight:  bool,
}

impl Slot {
    fn new(blocks: usize) -> Result<Self> {
        Ok(Self {
            stream:            Stream::new(StreamFlags::NON_BLOCKING, None)?,
            best_and_bid_dev:  DeviceBuffer::zeroed(1)?,
            indices_dev:       DeviceBuffer::zeroed(blocks)?,
            best_and_bid_host: LockedBuffer::new(&0u64, 1)?,
            indices_host:      LockedBuffer::new(&0u64, blocks)?,
            base_index: 0,
            in_flight:  false,
        })
    }

    fn launch(
        &mut self,
        func:    &cust::function::Function,
        seed:    u64,
        base:    u64,
        count:   u32,
        chunk:   u32,
        blocks:  u32,
        threads: u32,
    ) -> Result<()> {
        debug_assert!(!self.in_flight);
        self.base_index = base;
        self.best_and_bid_dev.set_8(0)?;
        let stream = &self.stream;
        unsafe {
            launch!(
                func<<<blocks, threads, 0, stream>>>(
                    seed, base, count, chunk,
                    self.best_and_bid_dev.as_device_ptr(),
                    self.indices_dev.as_device_ptr()
                )
            )?;
            self.best_and_bid_dev.async_copy_to(&mut self.best_and_bid_host[..], stream)?;
            self.indices_dev     .async_copy_to(&mut self.indices_host     [..], stream)?;
        }
        self.in_flight = true;
        Ok(())
    }

    fn finish(&mut self, fallback_base: u64) -> Result<(u32, u64)> {
        self.stream.synchronize()?;
        self.in_flight = false;
        let packed   = self.best_and_bid_host[0];
        let score    = (packed >> 32) as u32;
        let block_id = (packed & 0xffff_ffff) as usize;
        if score == 0 {
            return Ok((0, fallback_base));
        }
        let best_index = self.indices_host[block_id];
        Ok((score, best_index))
    }

    fn ensure_idle(&mut self, fallback_base: u64) -> Result<Option<(u32, u64)>> {
        if self.in_flight { Ok(Some(self.finish(fallback_base)?)) } else { Ok(None) }
    }
}

pub struct CudaWorker {
    slots:          Vec<Slot>,
    module:         Module,
    pub blocks:     u32,
    pub threads:    u32,
    pub chunk_size: u32,
    _ctx:           Context,
}

impl Drop for CudaWorker {
    fn drop(&mut self) {
        let _ = CurrentContext::set_current(&self._ctx);
    }
}

impl CudaWorker {
    pub fn new(blocks: u32, threads: u32, chunk_size: u32) -> Result<Self> {
        Self::new_with_slots(blocks, threads, chunk_size, 3)
    }

    pub fn new_with_slots(
        blocks: u32,
        threads: u32,
        chunk_size: u32,
        n_slots: usize,
    ) -> Result<Self> {
        eprintln!("[cuda] init");
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(0).context("no CUDA device found")?;
        let device_name = device.name().unwrap_or_else(|_| "<unknown>".to_string());
        let ctx    = Context::new(device)?;
        CurrentContext::set_current(&ctx)?;
        let module = Module::from_ptx(PTX, &[])?;
        let slots = (0..n_slots).map(|_| Slot::new(blocks as usize)).collect::<Result<Vec<_>>>()?;
        eprintln!("[cuda] ready — {}-buffered, {} blocks", n_slots, blocks);
        eprintln!("[cuda] device {device_name}");
        Ok(Self { slots, module, blocks, threads, chunk_size, _ctx: ctx })
    }

    pub fn run_batch(&mut self, seed: u64, base_index: u64, count: u32) -> Result<(u32, u64)> {
        CurrentContext::set_current(&self._ctx)?;
        let func   = self.module.get_function("bogo_shuffle_kernel")?;
        let stride = (self.blocks as u64) * (self.threads as u64) * (self.chunk_size as u64);
        let total  = count as u64;
        if total == 0 {
            return Ok((0, base_index));
        }
        let num_slots = self.slots.len();
        let mut best_c   = 0u32;
        let mut best_i   = base_index;
        let mut next_slot: usize = 0;
        let mut offset:    u64   = 0;
        let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
        loop {
            if offset < total && queue.len() < num_slots {
                let slot_idx  = next_slot % num_slots;
                let sub_count = stride.min(total - offset) as u32;
                let sub_base  = base_index + offset;
                if let Some((bc, bi)) = self.slots[slot_idx].ensure_idle(base_index)? {
                    if bc > best_c { best_c = bc; best_i = bi; }
                    queue.retain(|&s| s != slot_idx);
                }
                self.slots[slot_idx].launch(&func, seed, sub_base, sub_count,
                    self.chunk_size, self.blocks, self.threads)?;
                queue.push_back(slot_idx);
                next_slot += 1;
                offset    += sub_count as u64;
                continue;
            }
            if let Some(slot_idx) = queue.pop_front() {
                let (bc, bi) = self.slots[slot_idx].finish(base_index)?;
                if bc > best_c { best_c = bc; best_i = bi; }
            }
            if offset >= total && queue.is_empty() { break; }
        }
        Ok((best_c, best_i))
    }
}
