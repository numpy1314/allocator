use allocator::{BaseAllocator, PageAllocator, BitmapPageAllocator, AllocError};

const PAGE_SIZE: usize = 4096; // 4KB page size
const MEM_BASE: usize = 0x1000; // Memory pool start address
const MEM_SIZE: usize = 8 * 1024 * 1024; // 8MB memory pool

#[test]
fn test_multi_page_allocation() -> Result<(), AllocError> {
    let mut allocator = BitmapPageAllocator::<PAGE_SIZE>::new();
    allocator.init(MEM_BASE, MEM_SIZE);
    
    // Allocate 4 consecutive pages (16KB)
    let num_pages = 4;
    let addr = allocator.alloc_pages(num_pages, PAGE_SIZE)?;
    println!("[Multi-page allocation] Start address: 0x{:X}, Number of pages: {}", addr, num_pages);
    
    // Verify address alignment and continuity
    assert!(addr % PAGE_SIZE == 0, "Address is not aligned");
    assert!(addr >= MEM_BASE && addr < MEM_BASE + MEM_SIZE, "Address out of bounds");
    
    allocator.dealloc_pages(addr, num_pages);
    Ok(())
}

#[test]
fn test_specific_address_allocation() -> Result<(), AllocError> {
    let mut allocator = BitmapPageAllocator::<PAGE_SIZE>::new();
    allocator.init(MEM_BASE, MEM_SIZE);
    
    // Ensure the address meets 2MB alignment 
    let align = 2 * 1024 * 1024; // 2MB alignment
    let target_addr = (MEM_BASE + align - 1) & !(align - 1); // Align to the nearest 2MB boundary
    
    let num_pages = 2;
    
    // Allocate 2 pages at the specified address
    let addr = allocator.alloc_pages_at(target_addr, num_pages, align)?;
    println!("[Allocate at specified address] Requested address: 0x{:X}, Actual address: 0x{:X}", target_addr, addr);
    
    // Verify that the addresses match exactly
    assert_eq!(addr, target_addr, "Allocated address does not match the request");
    assert!(addr % align == 0, "Address does not meet alignment requirements");
    
    allocator.dealloc_pages(addr, num_pages);
    Ok(())
}