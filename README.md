# allocator

[![Build Status](https://github.com/arceos-org/allocator/actions/workflows/ci.yml/badge.svg)](https://github.com/arceos-org/allocator/actions)
[![crates.io](https://img.shields.io/crates/v/allocator.svg)](https://crates.io/crates/allocator)
[![docs.rs](https://img.shields.io/docsrs/allocator)](https://docs.rs/allocator)
[![License](https://img.shields.io/crates/l/allocator)](https://github.com/arceos-org/allocator/blob/main/LICENSE)


**A multi-algorithm memory allocator library with unified interfaces**​​, supporting page-level, byte-level memory allocation and unique ID allocation. Designed for scenarios requiring custom memory management like OS kernels and embedded systems.

## Core Features
- **Multi-algorithm Support​**​: Implements classic allocation algorithms including Bitmap (page-level), TLSF, Slab, and Buddy
​- **​Unified Interface**​​: Defines common operations through ByteAllocator/PageAllocator/IdAllocator traits
- **​​Configurability​**​: Flexible algorithm and memory capacity selection via Cargo features (e.g., page-alloc-1t supports 1TB page-level memory)
​​- **Standard Library Integration**​​: Supports core::alloc::Allocator trait for direct use with Vec/BTreeMap containers

## Example
### Example 1：TLSF Byte Allocator (General Memory Management)
```rust
use allocator::TlsfByteAllocator;
use core::alloc::Layout;

// Initialize TLSF allocator managing 1MB memory (starting at 0x100000)
let mut allocator = TlsfByteAllocator::new();
allocator.init(0x100000, 1024 * 1024);  // init(start_addr, size)

// Allocate 1024 bytes with 16-byte alignment
let layout = Layout::from_size_align(1024, 16).unwrap();
let ptr = allocator.alloc(layout).expect("分配失败");

// Simulate memory usage (write data to starting address)
unsafe { ptr.as_ptr().cast::<u32>().write(0xdead_beef) };

// Deallocate memory
allocator.dealloc(ptr, layout);
```

### Example 2: Bitmap Page Allocator (Kernel Page Management)
```rust
use allocator::BitmapPageAllocator;

// Define page size as 4KB (common kernel page size)
const PAGE_SIZE: usize = 4096;
let mut allocator = BitmapPageAllocator::<PAGE_SIZE>::new();

// Initialize: Manage 2GB memory starting at 0x200000 (2GB / 4KB = 524,288 pages)
allocator.init(0x200000, 2 * 1024 * 1024 * 1024);

// Allocate 10 contiguous pages (alignment requirement = page size)
let page_addr = allocator.alloc_pages(10, PAGE_SIZE).expect("页分配失败");
assert!(page_addr % PAGE_SIZE == 0);  // Ensure page-aligned address

// Deallocate the 10 pages
allocator.dealloc_pages(page_addr, 10);
```

## Feature Support
- bitmap: Enables Bitmap page allocator (default enables page-alloc-256m)
- tlsf: Enables TLSF byte allocator
- slab: Enables Slab byte allocator
- buddy: Enables Buddy byte allocator
- allocator_api: Enables standard library Allocator trait integration


## License
This project uses a multi-license model. Users may choose one of the following licenses:

GPL-3.0-or-later (GNU General Public License v3.0 or later)
Apache-2.0 (Apache License 2.0)
MulanPSL-2.0 (Mulan Permissive Software License, Version 2)

Full license texts are available [LICENSE File](https://github.com/arceos-org/allocator/blob/main/LICENSE)。