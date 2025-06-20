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

    // Fragmentation test
    #[test]
    fn fragmentation_resistance() {
        let (alloc, ptr, layout) = setup_allocator();
        
        // Phase 1: Allocate various size blocks
        let mut blocks = vec![];
        for i in 0..100 {
            let size = if i % 3 == 0 {
                32
            } else if i % 3 == 1 {
                64
            } else {
                128
            };
            
            let alloc_layout = Layout::from_size_align(size, 8).unwrap();
            let block = alloc.allocate(alloc_layout).expect("Allocation failed");
            blocks.push((block, alloc_layout, i));
        }
        
        // Phase 2: Free some blocks to create fragmentation
        blocks.retain(|(_, _, i)| !(i % 5 == 0 || i % 5 == 2));
        
        // Phase 3: Try to allocate large block
        let large_layout = Layout::from_size_align(8192, 8).unwrap();
        let large_block = alloc.allocate(large_layout);
        assert!(large_block.is_ok(), "Failed to allocate large block despite fragmentation");
        
        if let Ok(block) = large_block {
            unsafe { alloc.deallocate(block.cast(), large_layout) };
        }
        
        // Cleanup remaining blocks
        for (block, alloc_layout, _) in blocks {
            unsafe { alloc.deallocate(block.cast(), alloc_layout) };
        }
        
        unsafe {
            std::alloc::dealloc(ptr, layout);
        }
    }
    
    // Alignment test
    #[test]
    fn alignment_handling() {
        let (alloc, ptr, layout) = setup_allocator();
        
        for align_exp in 4..=12 {
            let align = 1 << align_exp; // 16, 32, ... 4096
            let test_layout = Layout::from_size_align(64, align).unwrap();
            
            let block = alloc.allocate(test_layout).expect("Alignment allocation failed");
            let addr = block.as_ptr() as *mut u8 as usize;
            
            assert_eq!(
                addr % align,
                0,
                "Misaligned block: address {:#x} not aligned to {}",
                addr, align
            );
            
            unsafe { alloc.deallocate(block.cast(), test_layout) };
        }
        
        unsafe {
            std::alloc::dealloc(ptr, layout);
        }
    }
}