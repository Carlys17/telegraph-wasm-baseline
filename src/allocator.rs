//! Global allocator and panic handler for `wasm32-unknown-unknown`.
//!
//! `no_std` targets require an explicit allocator. We use `dlmalloc` which
//! is small, well-tested, and has a proven WASM track record.

use dlmalloc::GlobalDlmalloc;

#[global_allocator]
static ALLOCATOR: GlobalDlmalloc = GlobalDlmalloc;

/// Panic handler — required for `no_std`.
/// Uses `unreachable_unchecked` which is safe under `panic = "abort"` profile.
/// On WASM this traps as unreachable; on native it's a no-op (the abort
/// profile terminates the process on actual panic before reaching here).
#[cfg(not(test))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}
