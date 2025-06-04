#![feature(allocator_api)]
#![feature(btreemap_alloc)]

use std::alloc::{Allocator, Layout};
use allocator::{AllocatorRc, SlabByteAllocator, BaseAllocator};

const POOL_SIZE: usize = 1024 * 1024 * 64; // 64MB内存池

// 创建页面对齐的内存池（返回原始指针和布局）
fn create_aligned_pool(size: usize, align: usize) -> (*mut u8, Layout) {
    assert!(align.is_power_of_two(), "对齐值必须是2的幂");
    let layout = Layout::from_size_align(size, align).expect("无效的内存布局");
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    (ptr, layout)
}

// 初始化分配器（返回分配器和内存池引用）
fn setup_allocator() -> (AllocatorRc<SlabByteAllocator>, *mut u8, Layout) {
    let (ptr, layout) = create_aligned_pool(POOL_SIZE, 4096);
    let heap_start = ptr as usize;
    
    // 验证内存池对齐
    assert_eq!(
        heap_start % 4096, 0,
        "内存池未对齐: 地址0x{:X}, 偏移量{}",
        heap_start, heap_start % 4096
    );
    
    // 初始化Slab分配器
    let mut slab_alloc = SlabByteAllocator::new();
    unsafe {
        slab_alloc.init(heap_start, POOL_SIZE);
    }
    
    // 创建分配器Rc封装
    let alloc = AllocatorRc::new(slab_alloc, unsafe {
        std::slice::from_raw_parts_mut(ptr, POOL_SIZE)
    });
    
    (alloc, ptr, layout)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // 1. 基本分配测试
    #[test]
    fn test_basic_allocation() {
        let (alloc, ptr, layout) = setup_allocator();
        
        // 测试分配
        let test_layout = Layout::new::<u32>();
        let test_ptr = alloc.allocate(test_layout).expect("分配失败");
        
        // 写入并验证数据
        unsafe {
            *(test_ptr.as_ptr() as *mut u32) = 0xDEADBEEF;
            assert_eq!(*(test_ptr.as_ptr() as *mut u32), 0xDEADBEEF, "数据验证失败");
        }
        
        // 释放测试内存
        unsafe {
            alloc.deallocate(test_ptr.cast(), test_layout);
        }
        
        // 释放内存池（测试结束后）
        unsafe {
            std::alloc::dealloc(ptr, layout);
        }
    }
}