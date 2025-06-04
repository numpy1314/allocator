use allocator::{BaseAllocator, PageAllocator, BitmapPageAllocator, AllocError};

const PAGE_SIZE: usize = 4096;

#[test] 
fn test_page_allocation() -> Result<(), AllocError> {
    let mut allocator = BitmapPageAllocator::<PAGE_SIZE>::new();
    allocator.init(0x1000, PAGE_SIZE);
    
    let addr = allocator.alloc_pages(1, PAGE_SIZE)?;
    println!("Allocated page at: 0x{:X}", addr);
    
    allocator.dealloc_pages(addr, 1);
    Ok(())
}