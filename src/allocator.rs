//! Global allocator and panic handler for `wasm32-unknown-unknown`.
//!
//! `no_std` targets require an explicit allocator. We use `dlmalloc` which
//! is small, well-tested, and has a proven WASM track record.
//!
//! Under the `bench` feature the crate is compiled NATIVELY (host target,
//! still `no_std` but linking `alloc`) for the measurement harness — we swap
//! in the std-backed allocator and a host panic handler. The bench build is
//! never shipped as WASM.

#[cfg(not(feature = "bench"))]
use dlmalloc::GlobalDlmalloc;

#[cfg(not(feature = "bench"))]
#[global_allocator]
static ALLOCATOR: GlobalDlmalloc = GlobalDlmalloc;

#[cfg(feature = "bench")]
#[global_allocator]
static ALLOCATOR: std::alloc::System = std::alloc::System;

/// Panic handler for the `no_std` WASM target — traps as unreachable.
#[cfg(not(any(test, feature = "bench")))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}
