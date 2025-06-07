#![feature(allocator_api)]
#![feature(btreemap_alloc)]

use std::alloc::{Allocator, Layout};
use allocator::{AllocatorRc, TlsfByteAllocator, BaseAllocator, ByteAllocator};

// 64MB memory pool
const POOL_SIZE: usize = 1024 * 1024 * 64;

// Create a page-aligned memory pool
fn create_aligned_pool(size: usize, align: usize) -> (*mut u8, Layout) {
    assert!(align.is_power_of_two(), "Alignment value must be a power of two");
    let layout = Layout::from_size_align(size, align).expect("Invalid memory layout");
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    (ptr, layout)
}

// Initialize the TLSF allocator
fn setup_allocator() -> (AllocatorRc<TlsfByteAllocator>, *mut u8, Layout) {
    let (ptr, layout) = create_aligned_pool(POOL_SIZE, 4096);
    let heap_start = ptr as usize;

    // Verify memory pool alignment
    assert_eq!(
        heap_start % 4096, 0,
        "Memory pool is not aligned: address 0x{:X}, offset {}",
        heap_start, heap_start % 4096
    );

    // Initialize the TLSF allocator
    let mut tlsf_alloc = TlsfByteAllocator::new();
    tlsf_alloc.init(heap_start, POOL_SIZE);

    // Create an allocator Rc wrapper
    let alloc = AllocatorRc::new(tlsf_alloc, unsafe {
        std::slice::from_raw_parts_mut(ptr, POOL_SIZE)
    });

    (alloc, ptr, layout)
}

#[cfg(test)]
mod tests {
    use super::*;

    // 1. Basic allocation test
    #[test]
    fn test_basic_allocation() {
        let (alloc, ptr, layout) = setup_allocator();

        // Test allocation
        let test_layout = Layout::new::<u32>();
        let test_ptr = alloc.allocate(test_layout).expect("Allocation failed");

        // Write and verify data
        unsafe {
            *(test_ptr.as_ptr() as *mut u32) = 0xDEADBEEF;
            assert_eq!(*(test_ptr.as_ptr() as *mut u32), 0xDEADBEEF, "Data verification failed");
        }

        // Free the test memory
        unsafe {
            alloc.deallocate(test_ptr.cast(), test_layout);
        }

        // Free the memory pool (after the test ends)
        unsafe {
            std::alloc::dealloc(ptr, layout);
        }
    }
}