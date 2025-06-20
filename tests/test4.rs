#![feature(allocator_api)]
#![feature(btreemap_alloc)]

use std::alloc::{Allocator, Layout};
use std::ptr::NonNull;
use allocator::{AllocatorRc, SlabByteAllocator, BaseAllocator};

// 64MB memory pool
const POOL_SIZE: usize = 1024 * 1024 * 64;

// Create a page-aligned memory pool (return raw pointer and layout)
fn create_aligned_pool(size: usize, align: usize) -> (*mut u8, Layout) {
    assert!(align.is_power_of_two(), "Alignment value must be a power of two");
    let layout = Layout::from_size_align(size, align).expect("Invalid memory layout");
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    (ptr, layout)
}

// Initialize the allocator (return the allocator and memory pool reference)
fn setup_allocator() -> (AllocatorRc<SlabByteAllocator>, *mut u8, Layout) {
    let (ptr, layout) = create_aligned_pool(POOL_SIZE, 4096);
    let heap_start = ptr as usize;

    // Verify memory pool alignment
    assert_eq!(
        heap_start % 4096, 0,
        "Memory pool is not aligned: address 0x{:X}, offset {}",
        heap_start, heap_start % 4096
    );

    // Initialize the Slab allocator
    let mut slab_alloc = SlabByteAllocator::new();
    unsafe {
        slab_alloc.init(heap_start, POOL_SIZE);
    }

    // Create an allocator Rc wrapper
    let alloc = AllocatorRc::new(slab_alloc, unsafe {
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

        // Free the memory test
        unsafe {
            alloc.deallocate(test_ptr.cast(), test_layout);
        }

        // Free the memory pool (after the test ends)
        unsafe {
            std::alloc::dealloc(ptr, layout);
        }
    }

    // 2. 测试Slab分配器的高效重用特性
    #[test]
    fn test_slab_reuse() {
        let (alloc, ptr, layout) = setup_allocator();
        let mut allocations = Vec::new();
        let test_layout = Layout::new::<usize>();
        
        // 分配128个对象
        for i in 0..128 {
            let block = alloc.allocate(test_layout).expect(&format!("Allocation {} failed", i));
            unsafe {
                // 通过原始指针访问内存
                let raw_ptr = block.as_ptr() as *mut usize;
                *raw_ptr = i;
            }
            allocations.push(block);
        }
        
        // 释放所有奇数索引的对象
        for (i, block) in allocations.iter_mut().enumerate().filter(|(i, _)| i % 2 == 1) {
            unsafe {
                // 转换为NonNull<u8>进行释放
                alloc.deallocate(
                    NonNull::new(block.as_ptr() as *mut u8).unwrap(), 
                    test_layout
                );
            }
            *block = NonNull::slice_from_raw_parts(NonNull::dangling(), 0); // 标记为已释放
        }
        
        // 分配32个新对象
        let mut new_allocations = Vec::new();
        for i in 0..32 {
            let block = alloc.allocate(test_layout).expect(&format!("New allocation {} failed", i));
            unsafe {
                let raw_ptr = block.as_ptr() as *mut usize;
                *raw_ptr = i + 1000;
            }
            new_allocations.push(block);
        }
        
        // 验证所有分配都在内存池范围内
        let start = ptr as usize;
        let end = start + POOL_SIZE;
        for block in &allocations {
            if block.as_ptr() as *const u8 != NonNull::dangling().as_ptr() {
                // 正确获取地址值
                let addr = block.as_ptr() as *const u8 as usize;
                assert!(
                    addr >= start && addr < end, 
                    "Allocation outside pool: 0x{:X} (start: 0x{:X}, end: 0x{:X})", 
                    addr, start, end
                );
            }
        }
        
        // 释放所有剩余对象
        unsafe {
            for block in allocations.iter().chain(&new_allocations) {
                if block.as_ptr() as *const u8 != NonNull::dangling().as_ptr() {
                    // 正确转换指针类型进行释放
                    alloc.deallocate(
                        NonNull::new(block.as_ptr() as *mut u8).unwrap(),
                        test_layout
                    );
                }
            }
        }
        
        unsafe {
            std::alloc::dealloc(ptr, layout);
        }
    }

    // 3. 测试分配器统计信息（通过间接方式验证）
    #[test]
    fn test_allocator_stats() {
        let (alloc, ptr, layout) = setup_allocator();
        
        // 初始内存状态应该为空闲
        let test_layout = Layout::new::<u32>();
        let test_ptr = alloc.allocate(test_layout).expect("Initial allocation failed");
        unsafe {
            alloc.deallocate(NonNull::new(test_ptr.as_ptr() as *mut u8).unwrap(), test_layout);
        }
        
        // 进行一些分配
        let layouts = [
            Layout::new::<u8>(),
            Layout::new::<u32>(),
            Layout::new::<u64>(),
            Layout::array::<usize>(16).unwrap(),
            Layout::array::<usize>(256).unwrap(),
        ];
        
        let mut allocations = Vec::new();
        for (i, layout) in layouts.iter().enumerate() {
            for _ in 0..(10 * (i + 1)) {
                let block = alloc.allocate(*layout).expect(&format!("Allocation failed for layout index {}", i));
                allocations.push((block, *layout));
            }
        }
        
        // 释放部分内存
        for (i, (block, layout)) in allocations.iter().enumerate().filter(|(i, _)| i % 3 == 0) {
            unsafe {
                alloc.deallocate(NonNull::new(block.as_ptr() as *mut u8).unwrap(), *layout);
            }
        }
        
        // 分配一个比剩余空间大的内存块应该失败
        let huge_layout = Layout::array::<u8>(POOL_SIZE * 2).unwrap();
        let result = alloc.allocate(huge_layout);
        assert!(result.is_err(), "Allocation of impossible size succeeded");
        
        // 清理
        for (i, (block, layout)) in allocations.iter().enumerate().filter(|(i, _)| i % 3 != 0) {
            unsafe {
                alloc.deallocate(NonNull::new(block.as_ptr() as *mut u8).unwrap(), *layout);
            }
        }
        
        unsafe {
            std::alloc::dealloc(ptr, layout);
        }
    }
}