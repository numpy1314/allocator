use allocator::{BaseAllocator, PageAllocator, BitmapPageAllocator, AllocError};

const PAGE_SIZE: usize = 4096; // 4KB页大小
const MEM_BASE: usize = 0x1000; // 内存池起始地址
const MEM_SIZE: usize = 8 * 1024 * 1024; // 8MB内存池

#[test]
fn test_multi_page_allocation() -> Result<(), AllocError> {
    let mut allocator = BitmapPageAllocator::<PAGE_SIZE>::new();
    allocator.init(MEM_BASE, MEM_SIZE);
    
    // 分配连续4页（16KB）
    let num_pages = 4;
    let addr = allocator.alloc_pages(num_pages, PAGE_SIZE)?;
    println!("[多页分配] 起始地址: 0x{:X}, 页数: {}", addr, num_pages);
    
    // 验证地址对齐和连续性
    assert!(addr % PAGE_SIZE == 0, "地址未对齐");
    assert!(addr >= MEM_BASE && addr < MEM_BASE + MEM_SIZE, "地址越界");
    
    allocator.dealloc_pages(addr, num_pages);
    Ok(())
}

#[test]
fn test_specific_address_allocation() -> Result<(), AllocError> {
    let mut allocator = BitmapPageAllocator::<PAGE_SIZE>::new();
    allocator.init(MEM_BASE, MEM_SIZE);
    
    // 确保地址满足2MB对齐（关键修正）
    let align = 2 * 1024 * 1024; // 2MB对齐
    let target_addr = (MEM_BASE + align - 1) & !(align - 1); // 对齐到最近的2MB边界
    
    let num_pages = 2;
    
    // 在指定地址分配2页
    let addr = allocator.alloc_pages_at(target_addr, num_pages, align)?;
    println!("[指定地址分配] 请求地址: 0x{:X}, 实际地址: 0x{:X}", target_addr, addr);
    
    // 验证地址精确匹配
    assert_eq!(addr, target_addr, "分配地址与请求不匹配");
    assert!(addr % align == 0, "地址未满足对齐要求");
    
    allocator.dealloc_pages(addr, num_pages);
    Ok(())
}