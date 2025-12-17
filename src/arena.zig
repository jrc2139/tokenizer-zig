//! TokenizerArena - Pre-allocated memory pool for zero-allocation tokenization
//!
//! This module provides thread-local arenas that contain all the memory needed
//! for tokenization. After initialization, encode() operations perform zero
//! allocations - they just reset and reuse the pre-allocated buffers.

const std = @import("std");
const SpanToken = @import("token.zig").SpanToken;
const SpanEncoding = @import("encoding.zig").SpanEncoding;
const lib = @import("lib.zig");

// ============================================================================
// BPE Scratch Data Structures (for O(n log n) algorithm)
// ============================================================================

/// BPE symbol with linked-list pointers for O(1) removal during merge
pub const BPESymbol = struct {
    /// Token ID from vocabulary
    id: u32,
    /// Byte offset into input (start)
    start: u32,
    /// Byte offset into input (end)
    end: u32,
    /// Index of previous symbol (SENTINEL if none)
    prev: u16,
    /// Index of next symbol (SENTINEL if none)
    next: u16,

    pub const SENTINEL: u16 = std.math.maxInt(u16);

    /// Check if this symbol has been merged (removed from list)
    pub inline fn isRemoved(self: BPESymbol) bool {
        return self.prev == SENTINEL and self.next == SENTINEL and self.id == std.math.maxInt(u32);
    }

    /// Mark as removed
    pub inline fn markRemoved(self: *BPESymbol) void {
        self.id = std.math.maxInt(u32);
        self.prev = SENTINEL;
        self.next = SENTINEL;
    }
};

/// Entry in BPE merge priority heap
pub const BPEPairEntry = struct {
    /// Index of left symbol in symbol array
    left_idx: u16,
    /// Index of right symbol in symbol array
    right_idx: u16,
    /// Merge rank (lower = higher priority)
    rank: u32,
};

/// Min-heap for BPE merge pairs
pub const BPEPairHeap = struct {
    items: []BPEPairEntry,
    len: usize,
    capacity: usize,

    const Self = @This();

    pub fn init(buffer: []BPEPairEntry) Self {
        return .{
            .items = buffer,
            .len = 0,
            .capacity = buffer.len,
        };
    }

    pub fn reset(self: *Self) void {
        self.len = 0;
    }

    /// Insert an entry, maintaining heap property
    pub fn insert(self: *Self, entry: BPEPairEntry) void {
        if (self.len >= self.capacity) return;

        // Add at end
        self.items[self.len] = entry;
        var idx = self.len;
        self.len += 1;

        // Bubble up
        while (idx > 0) {
            const parent = (idx - 1) / 2;
            if (self.items[parent].rank <= self.items[idx].rank) break;
            const tmp = self.items[parent];
            self.items[parent] = self.items[idx];
            self.items[idx] = tmp;
            idx = parent;
        }
    }

    /// Pop the minimum element
    pub fn pop(self: *Self) ?BPEPairEntry {
        if (self.len == 0) return null;

        const result = self.items[0];
        self.len -= 1;

        if (self.len == 0) return result;

        // Move last to root and bubble down
        self.items[0] = self.items[self.len];
        var idx: usize = 0;

        while (true) {
            const left = 2 * idx + 1;
            const right = 2 * idx + 2;
            var smallest = idx;

            if (left < self.len and self.items[left].rank < self.items[smallest].rank) {
                smallest = left;
            }
            if (right < self.len and self.items[right].rank < self.items[smallest].rank) {
                smallest = right;
            }

            if (smallest == idx) break;

            const tmp = self.items[idx];
            self.items[idx] = self.items[smallest];
            self.items[smallest] = tmp;
            idx = smallest;
        }

        return result;
    }

    pub fn isEmpty(self: *const Self) bool {
        return self.len == 0;
    }
};

// ============================================================================
// TokenizerArena - Thread-Local Memory Pool
// ============================================================================

/// Configuration for TokenizerArena
pub const ArenaConfig = struct {
    /// Maximum input sequence length in bytes
    max_sequence_length: u32 = 8192,
    /// Maximum output tokens
    max_tokens: u32 = 512,
};

/// Pre-allocated memory arena for zero-allocation tokenization.
/// Each thread should have its own arena (via thread-local storage).
pub const TokenizerArena = struct {
    /// Allocator used for the arena's buffers
    allocator: std.mem.Allocator,

    /// Pre-allocated encoding output
    encoding: SpanEncoding,

    /// BPE symbol scratch space (one per input byte max)
    bpe_symbols: []BPESymbol,

    /// BPE merge heap scratch space
    bpe_heap_buffer: []BPEPairEntry,

    /// BPE heap wrapper
    bpe_heap: BPEPairHeap,

    /// Pre-tokenizer scratch space: (start, end) pairs for word boundaries
    pretoken_spans: [][2]u32,

    /// Number of pre-token spans in use
    pretoken_len: u32,

    /// Configuration
    config: ArenaConfig,

    const Self = @This();

    /// Initialize the arena with pre-allocated buffers.
    /// This is the only allocation point for this arena.
    pub fn init(allocator: std.mem.Allocator, config: ArenaConfig) !Self {
        var encoding = try SpanEncoding.init(allocator, config.max_tokens);
        errdefer encoding.deinit(allocator);

        // BPE needs one symbol per input character (Unicode codepoint ≤ 4 bytes)
        const max_symbols = config.max_sequence_length;
        const bpe_symbols = try allocator.alloc(BPESymbol, max_symbols);
        errdefer allocator.free(bpe_symbols);

        // Heap can have at most max_symbols - 1 pairs
        const bpe_heap_buffer = try allocator.alloc(BPEPairEntry, max_symbols);
        errdefer allocator.free(bpe_heap_buffer);

        // Pre-tokenizer: assume ~1 word per 5 characters on average
        const max_pretokens = config.max_sequence_length / 4;
        const pretoken_spans = try allocator.alloc([2]u32, max_pretokens);
        errdefer allocator.free(pretoken_spans);

        return .{
            .allocator = allocator,
            .encoding = encoding,
            .bpe_symbols = bpe_symbols,
            .bpe_heap_buffer = bpe_heap_buffer,
            .bpe_heap = BPEPairHeap.init(bpe_heap_buffer),
            .pretoken_spans = pretoken_spans,
            .pretoken_len = 0,
            .config = config,
        };
    }

    /// Free all pre-allocated buffers
    pub fn deinit(self: *Self) void {
        self.encoding.deinit(self.allocator);
        self.allocator.free(self.bpe_symbols);
        self.allocator.free(self.bpe_heap_buffer);
        self.allocator.free(self.pretoken_spans);
    }

    /// Reset arena for new tokenization (O(1), no allocation)
    pub fn reset(self: *Self, input: []const u8) void {
        self.encoding.reset(input);
        self.bpe_heap.reset();
        self.pretoken_len = 0;
    }

    /// Add a pre-token span
    pub fn addPretokenSpan(self: *Self, start: u32, end: u32) void {
        if (self.pretoken_len < self.pretoken_spans.len) {
            self.pretoken_spans[self.pretoken_len] = .{ start, end };
            self.pretoken_len += 1;
        }
    }

    /// Get pre-token spans
    pub fn getPretokenSpans(self: *const Self) []const [2]u32 {
        return self.pretoken_spans[0..self.pretoken_len];
    }

    /// Calculate total memory used by this arena
    pub fn memoryUsage(self: *const Self) usize {
        return @sizeOf(Self) +
            self.encoding.tokens.len * @sizeOf(SpanToken) +
            self.encoding.ids.len * @sizeOf(u32) * 4 + // ids, attention, types, offsets
            self.bpe_symbols.len * @sizeOf(BPESymbol) +
            self.bpe_heap_buffer.len * @sizeOf(BPEPairEntry) +
            self.pretoken_spans.len * @sizeOf([2]u32);
    }
};

// ============================================================================
// Thread-Local Arena Pool
// ============================================================================

/// Global arena pool for thread-local access
var arena_pool_mutex: std.Thread.Mutex = .{};
var arena_pool: ?*ArenaPool = null;

/// Pool of arenas for multi-threaded use
pub const ArenaPool = struct {
    allocator: std.mem.Allocator,
    config: ArenaConfig,
    arenas: std.ArrayListUnmanaged(*TokenizerArena),

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, config: ArenaConfig) !*Self {
        const pool = try allocator.create(Self);
        pool.* = .{
            .allocator = allocator,
            .config = config,
            .arenas = .{},
        };
        return pool;
    }

    pub fn deinit(self: *Self) void {
        for (self.arenas.items) |arena| {
            arena.deinit();
            self.allocator.destroy(arena);
        }
        self.arenas.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    /// Acquire an arena (creates new one if needed)
    pub fn acquire(self: *Self) !*TokenizerArena {
        const arena = try self.allocator.create(TokenizerArena);
        arena.* = try TokenizerArena.init(self.allocator, self.config);
        try self.arenas.append(self.allocator, arena);
        return arena;
    }
};

/// Thread-local arena pointer
threadlocal var tl_arena: ?*TokenizerArena = null;

/// Get thread-local arena, creating one if needed
pub fn getThreadLocalArena(pool: *ArenaPool) !*TokenizerArena {
    if (tl_arena) |arena| {
        return arena;
    }

    arena_pool_mutex.lock();
    defer arena_pool_mutex.unlock();

    const arena = try pool.acquire();
    tl_arena = arena;
    return arena;
}

/// Initialize the global arena pool
pub fn initGlobalPool(allocator: std.mem.Allocator, config: ArenaConfig) !void {
    arena_pool_mutex.lock();
    defer arena_pool_mutex.unlock();

    if (arena_pool != null) return;
    arena_pool = try ArenaPool.init(allocator, config);
}

/// Deinitialize the global arena pool
pub fn deinitGlobalPool() void {
    arena_pool_mutex.lock();
    defer arena_pool_mutex.unlock();

    if (arena_pool) |pool| {
        pool.deinit();
        arena_pool = null;
    }
    tl_arena = null;
}

/// Get thread-local arena from global pool
pub fn getArena() !*TokenizerArena {
    if (arena_pool) |pool| {
        return getThreadLocalArena(pool);
    }
    return error.ArenaPoolNotInitialized;
}

// ============================================================================
// Unit Tests
// ============================================================================

test "TokenizerArena init and deinit" {
    const allocator = std.testing.allocator;

    var arena = try TokenizerArena.init(allocator, .{
        .max_sequence_length = 1024,
        .max_tokens = 128,
    });
    defer arena.deinit();

    try std.testing.expectEqual(@as(u32, 128), arena.encoding.capacity);
    try std.testing.expect(arena.bpe_symbols.len >= 1024);
}

test "TokenizerArena reset" {
    const allocator = std.testing.allocator;

    var arena = try TokenizerArena.init(allocator, .{});
    defer arena.deinit();

    const input = "hello world";
    arena.reset(input);

    try std.testing.expectEqual(@as(u32, 0), arena.encoding.len);
    try std.testing.expectEqualStrings(input, arena.encoding.input);
}

test "TokenizerArena pretoken spans" {
    const allocator = std.testing.allocator;

    var arena = try TokenizerArena.init(allocator, .{});
    defer arena.deinit();

    arena.reset("hello world test");
    arena.addPretokenSpan(0, 5); // "hello"
    arena.addPretokenSpan(6, 11); // "world"
    arena.addPretokenSpan(12, 16); // "test"

    const spans = arena.getPretokenSpans();
    try std.testing.expectEqual(@as(usize, 3), spans.len);
    try std.testing.expectEqual(@as(u32, 0), spans[0][0]);
    try std.testing.expectEqual(@as(u32, 5), spans[0][1]);
}

test "BPEPairHeap min heap property" {
    var buffer: [10]BPEPairEntry = undefined;
    var heap = BPEPairHeap.init(&buffer);

    // Insert in random order
    heap.insert(.{ .left_idx = 0, .right_idx = 1, .rank = 5 });
    heap.insert(.{ .left_idx = 1, .right_idx = 2, .rank = 2 });
    heap.insert(.{ .left_idx = 2, .right_idx = 3, .rank = 8 });
    heap.insert(.{ .left_idx = 3, .right_idx = 4, .rank = 1 });
    heap.insert(.{ .left_idx = 4, .right_idx = 5, .rank = 3 });

    // Should pop in sorted order (min first)
    try std.testing.expectEqual(@as(u32, 1), heap.pop().?.rank);
    try std.testing.expectEqual(@as(u32, 2), heap.pop().?.rank);
    try std.testing.expectEqual(@as(u32, 3), heap.pop().?.rank);
    try std.testing.expectEqual(@as(u32, 5), heap.pop().?.rank);
    try std.testing.expectEqual(@as(u32, 8), heap.pop().?.rank);
    try std.testing.expect(heap.pop() == null);
}

test "BPEPairHeap reset" {
    var buffer: [10]BPEPairEntry = undefined;
    var heap = BPEPairHeap.init(&buffer);

    heap.insert(.{ .left_idx = 0, .right_idx = 1, .rank = 5 });
    heap.insert(.{ .left_idx = 1, .right_idx = 2, .rank = 2 });

    try std.testing.expectEqual(@as(usize, 2), heap.len);

    heap.reset();
    try std.testing.expectEqual(@as(usize, 0), heap.len);
    try std.testing.expect(heap.isEmpty());
}

test "BPESymbol sentinel" {
    const sym = BPESymbol{
        .id = 0,
        .start = 0,
        .end = 5,
        .prev = BPESymbol.SENTINEL,
        .next = 1,
    };
    try std.testing.expect(!sym.isRemoved());

    var sym2 = sym;
    sym2.markRemoved();
    try std.testing.expect(sym2.isRemoved());
}

test "ArenaPool basic" {
    const allocator = std.testing.allocator;

    var pool = try ArenaPool.init(allocator, .{});
    defer pool.deinit();

    const arena = try pool.acquire();
    _ = arena;

    try std.testing.expectEqual(@as(usize, 1), pool.arenas.items.len);
}
