extern crate alloc;

use alloc::boxed::Box;
use allocator::BuddyByteAllocator;
use core::alloc::Layout;
use allocator::ByteAllocator;
use allocator::BaseAllocator;

// Create a heap memory pool (to avoid stack overflow)
fn create_test_pool(size: usize) -> Box<[u8]> {
    vec![0u8; size].into_boxed_slice()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fragmentation_handling() {
        const HEAP_SIZE: usize = 4 * 1024 * 1024; // 4MB
        let heap_mem = create_test_pool(HEAP_SIZE);
        let heap_start = heap_mem.as_ptr() as usize;
        let mut allocator = BuddyByteAllocator::new();
        
        unsafe {
            allocator.init(heap_start, heap_mem.len());
        }

        // Allocate multiple small blocks (to create fragmentation)
        let mut ptrs = Vec::new();
        let small_layout = Layout::from_size_align(4096, 4096).unwrap(); // 4KB
        
        for _ in 0..100 {
            let ptr = allocator.alloc(small_layout).expect("Failed to allocate small block");
            ptrs.push(ptr);
        }
        
        // Free all blocks at odd indices
        for i in (1..ptrs.len()).step_by(2) {
            allocator.dealloc(ptrs[i], small_layout);
        }
        
        // Try to allocate a large block (should be able to utilize merged fragments)
        let large_size = 1024 * 1024; // 1MB
        let large_layout = Layout::from_size_align(large_size, large_size).unwrap();
        assert!(
            allocator.alloc(large_layout).is_ok(),
            "Should be able to allocate a {} - byte large block using fragments",
            large_size
        );
        
        // Clean up the remaining memory
        for i in (0..ptrs.len()).step_by(2) {
            allocator.dealloc(ptrs[i], small_layout);
        }
    }
}