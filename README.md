### 项目概述

该项目实现了多种内存分配算法，通过统一的接口将这些算法封装起来，方便开发者根据不同的需求选择合适的分配器。项目支持多种特性开关，可根据需要启用不同的分配器。

### 项目结构和模块

- **`src/lib.rs`**: 项目的核心文件，定义了统一的分配器接口，包括`BaseAllocator`、字节粒度分配器、页粒度分配器和唯一 ID 分配器的接口。同时，提供了一些辅助函数，如地址对齐和检查对齐的函数。此外，还实现了 `AllocatorRc` 结构体，用于将字节分配器包装成 `core::alloc::Allocator` 类型。
- **`src/bitmap.rs`**: 实现了基于位图的页粒度内存分配器 `BitmapPageAllocator`。使用位图来跟踪每个页面的分配状态，支持不同大小的内存范围。
- **`src/buddy.rs`**: 实现了基于伙伴系统的字节粒度内存分配器 `BuddyByteAllocator`。使用 `buddy_system_allocator` 库来管理内存。
- **`src/slab.rs`**: 实现了基于 slab 分配器的字节粒度内存分配器 `SlabByteAllocator`。使用 `slab_allocator` 库来管理内存。
- **`src/tlsf.rs`**: 实现了基于 TLSF（Two-Level Segregated Fit）算法的字节粒度内存分配器 `TlsfByteAllocator`。使用 `rlsf` 库来管理内存。

### Feature

#### 项目中通过 `cfg` 特性开关来控制不同模块和功能的编译，以下是所有可用的特性开关及其作用

##### 1. **`bitmap`**

- **作用**：启用基于位图的页粒度内存分配器。
- **关联模块**：`bitmap.rs`
- **导出类型**：`BitmapPageAllocator`
- **功能**：使用位图数据结构管理页级内存分配，适合管理大块连续内存区域。

##### 2. **`buddy`**

- **作用**：启用基于伙伴系统的字节粒度内存分配器。
- **关联模块**：`buddy.rs`
- **导出类型**：`BuddyByteAllocator`
- **功能**：基于伙伴系统算法实现细粒度内存分配，适合需要处理不同大小内存块的场景。

##### 3. **`slab`**

- **作用**：启用基于 slab 分配器的字节粒度内存分配器。
- **关联模块**：`slab.rs`
- **导出类型**：`SlabByteAllocator`
- **功能**：针对特定大小对象的高效分配，减少内部碎片，适合频繁分配 / 释放相同大小对象的场景。

##### 4. **`tlsf`**

- **作用**：启用基于 TLSF（Two-Level Segregated Fit）算法的字节粒度内存分配器。
- **关联模块**：`tlsf.rs`
- **导出类型**：`TlsfByteAllocator`
- **功能**：结合多级空闲列表和位图索引，实现高效的内存分配与回收，尤其适合分配不同大小内存块的场景。

##### 5. **`allocator_api`**

- **作用**：启用与 Rust 标准库内存分配器接口的集成。
- **关联模块**：`allocator_api` 子模块
- **导出类型**：`AllocatorRc`
- **功能**：将自定义字节分配器（`ByteAllocator`）包装为实现 `core::alloc::Allocator` 特征的类型，允许在需要标准分配器接口的场景中使用自定义分配器（如 `Rc`、`Box` 等）。

### Trait

BaseAllocator（基础分配器）
│
├── ByteAllocator（字节粒度分配器）
│   ├─ 继承自 BaseAllocator
│   ├─ 核心方法：
│   │   ├─ alloc(Layout)       # 按内存布局分配字节[3,5](@ref)
│   │   └─ dealloc(NonNull<u8>, Layout)  # 释放指定布局的字节内存[3,5](@ref)
│   └─ 统计方法：
│       ├─ total_bytes()       # 总字节容量
│       ├─ used_bytes()        # 已使用字节数
│       └─ available_bytes()   # 剩余可用字节数
│
├── PageAllocator（页粒度分配器）
│   ├─ 继承自 BaseAllocator
│   ├─ 常量：
│   │   └─ const PAGE_SIZE: usize  # 页大小（单位：字节）[2](@ref)
│   ├─ 核心方法：
│   │   ├─ alloc_pages(num_pages, align_pow2)    # 分配指定页数（按对齐要求）[2](@ref)
│   │   ├─ dealloc_pages(pos, num_pages)         # 释放连续页[2](@ref)
│   │   └─ alloc_pages_at(...)                   # 在指定地址分配页[2](@ref)
│   └─ 统计方法：
│       ├─ total_pages()       # 总页数
│       ├─ used_pages()        # 已分配页数
│       └─ available_pages()   # 可用页数
│
└── IdAllocator（唯一ID分配器）
    ├─ 继承自 BaseAllocator
    ├─ 核心方法：
    │   ├─ alloc_id(count, align_pow2)    # 分配连续ID（按对齐要求）
    │   ├─ dealloc_id(start_id, count)    # 释放连续ID
    │   └─ alloc_fixed_id(id)             # 分配指定固定ID
    └─ 统计方法：
        ├─ size()        # 最大可分配ID数
        ├─ used()        # 已分配ID数
        └─ available()   # 可用ID数