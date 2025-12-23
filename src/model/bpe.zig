//! BPE (Byte Pair Encoding) tokenization model
//!
//! BPE is used by GPT-2, RoBERTa, and many other models.
//!
//! This module provides two implementations:
//! - `BPE`: Original implementation with allocations per tokenize() call
//! - `FastBPE`: Zero-allocation implementation using pre-allocated arenas

const std = @import("std");
const model = @import("model.zig");
const Token = @import("../token.zig").Token;
const SpanToken = @import("../token.zig").SpanToken;
const arena_mod = @import("../arena.zig");
const BPESymbol = arena_mod.BPESymbol;
const BPEPairHeap = arena_mod.BPEPairHeap;
const TokenizerArena = arena_mod.TokenizerArena;
const SpanEncoding = @import("../encoding.zig").SpanEncoding;

/// A pair of token IDs for BPE merging
pub const Pair = struct {
    first: u32,
    second: u32,

    pub fn hash(self: Pair) u64 {
        return @as(u64, self.first) << 32 | @as(u64, self.second);
    }
};

/// Value associated with a merge pair (rank and result ID)
pub const PairVal = struct {
    rank: u32,
    new_id: u32,
};

/// BPE tokenization model
pub const BPE = struct {
    allocator: std.mem.Allocator,
    vocab: std.StringHashMapUnmanaged(u32),
    vocab_r: std.AutoHashMapUnmanaged(u32, []const u8),
    merges: std.AutoHashMapUnmanaged(u64, PairVal), // hash(Pair) -> PairVal
    unk_token: ?[]const u8,
    continuing_subword_prefix: ?[]const u8,
    end_of_word_suffix: ?[]const u8,
    dropout: ?f32,
    owns_strings: bool, // Whether we own the config strings

    const Self = @This();

    pub const Config = struct {
        unk_token: ?[]const u8 = null,
        continuing_subword_prefix: ?[]const u8 = null,
        end_of_word_suffix: ?[]const u8 = null,
        dropout: ?f32 = null,
    };

    /// Initialize with borrowed strings (caller retains ownership)
    pub fn init(
        allocator: std.mem.Allocator,
        vocab: std.StringHashMapUnmanaged(u32),
        merges: std.AutoHashMapUnmanaged(u64, PairVal),
        config: Config,
    ) !Self {
        // Build reverse vocab
        var vocab_r = std.AutoHashMapUnmanaged(u32, []const u8){};
        var it = vocab.iterator();
        while (it.next()) |entry| {
            try vocab_r.put(allocator, entry.value_ptr.*, entry.key_ptr.*);
        }

        return .{
            .allocator = allocator,
            .vocab = vocab,
            .vocab_r = vocab_r,
            .merges = merges,
            .unk_token = config.unk_token,
            .continuing_subword_prefix = config.continuing_subword_prefix,
            .end_of_word_suffix = config.end_of_word_suffix,
            .dropout = config.dropout,
            .owns_strings = false,
        };
    }

    /// Initialize with owned strings (BPE takes ownership and frees on deinit)
    pub fn initOwned(
        allocator: std.mem.Allocator,
        vocab: std.StringHashMapUnmanaged(u32),
        merges: std.AutoHashMapUnmanaged(u64, PairVal),
        unk_token: ?[]const u8,
        prefix: ?[]const u8,
        suffix: ?[]const u8,
    ) !Self {
        // Build reverse vocab
        var vocab_r = std.AutoHashMapUnmanaged(u32, []const u8){};
        var it = vocab.iterator();
        while (it.next()) |entry| {
            try vocab_r.put(allocator, entry.value_ptr.*, entry.key_ptr.*);
        }

        return .{
            .allocator = allocator,
            .vocab = vocab,
            .vocab_r = vocab_r,
            .merges = merges,
            .unk_token = unk_token,
            .continuing_subword_prefix = prefix,
            .end_of_word_suffix = suffix,
            .dropout = null,
            .owns_strings = true,
        };
    }

    pub fn deinit(self: *Self) void {
        // Free owned config strings
        if (self.owns_strings) {
            if (self.unk_token) |s| self.allocator.free(s);
            if (self.continuing_subword_prefix) |s| self.allocator.free(s);
            if (self.end_of_word_suffix) |s| self.allocator.free(s);
        }

        var it = self.vocab.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.vocab.deinit(self.allocator);
        self.vocab_r.deinit(self.allocator);
        self.merges.deinit(self.allocator);
    }

    /// Get as generic Model interface
    pub fn getModel(self: *Self) model.Model {
        return .{
            .ptr = self,
            .tokenize_fn = tokenizeImpl,
            .token_to_id_fn = tokenToIdImpl,
            .id_to_token_fn = idToTokenImpl,
            .get_vocab_size_fn = getVocabSizeImpl,
            .deinit = deinitImpl,
            .destroy = destroyImpl,
        };
    }

    fn tokenizeImpl(ctx: *anyopaque, allocator: std.mem.Allocator, sequence: []const u8) anyerror![]Token {
        const self: *Self = @ptrCast(@alignCast(ctx));
        return self.tokenize(allocator, sequence);
    }

    fn tokenToIdImpl(ctx: *anyopaque, token: []const u8) ?u32 {
        const self: *Self = @ptrCast(@alignCast(ctx));
        return self.vocab.get(token);
    }

    fn idToTokenImpl(ctx: *anyopaque, id: u32) ?[]const u8 {
        const self: *Self = @ptrCast(@alignCast(ctx));
        return self.vocab_r.get(id);
    }

    fn getVocabSizeImpl(ctx: *anyopaque) usize {
        const self: *Self = @ptrCast(@alignCast(ctx));
        return self.vocab.count();
    }

    fn deinitImpl(ctx: *anyopaque) void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        self.deinit();
    }

    fn destroyImpl(allocator: std.mem.Allocator, ctx: *anyopaque) void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        allocator.destroy(self);
    }

    /// Tokenize using BPE algorithm
    pub fn tokenize(self: *Self, allocator: std.mem.Allocator, sequence: []const u8) ![]Token {
        if (sequence.len == 0) {
            return &.{};
        }

        // Start with character-level tokens
        var word = std.ArrayListUnmanaged(u32){};
        defer word.deinit(allocator);

        var char_offsets = std.ArrayListUnmanaged(struct { start: u32, end: u32 }){};
        defer char_offsets.deinit(allocator);

        var byte_idx: u32 = 0;
        var iter = std.unicode.Utf8Iterator{ .bytes = sequence, .i = 0 };
        while (iter.nextCodepointSlice()) |char_slice| {
            // TODO: Add prefix/suffix handling when needed
            const char_str = char_slice;
            const char_len: u32 = @intCast(char_slice.len);

            if (self.vocab.get(char_str)) |id| {
                try word.append(allocator, id);
                try char_offsets.append(allocator, .{
                    .start = byte_idx,
                    .end = byte_idx + char_len,
                });
            } else if (self.unk_token) |unk| {
                if (self.vocab.get(unk)) |unk_id| {
                    try word.append(allocator, unk_id);
                    try char_offsets.append(allocator, .{
                        .start = byte_idx,
                        .end = byte_idx + char_len,
                    });
                }
                // If unk_token exists but isn't in vocab, skip this character
            }
            // If no vocab match and no unk_token, skip this character

            byte_idx += char_len;
        }

        // Apply merges iteratively
        while (word.items.len > 1) {
            // Find the best pair to merge
            var best_pair: ?Pair = null;
            var best_rank: u32 = std.math.maxInt(u32);

            for (0..word.items.len - 1) |i| {
                const pair = Pair{
                    .first = word.items[i],
                    .second = word.items[i + 1],
                };
                if (self.merges.get(pair.hash())) |pair_val| {
                    if (pair_val.rank < best_rank) {
                        best_rank = pair_val.rank;
                        best_pair = pair;
                    }
                }
            }

            if (best_pair == null) {
                break; // No more merges possible
            }

            // Apply the merge
            const pair = best_pair.?;
            const pair_val = self.merges.get(pair.hash()).?;

            var i: usize = 0;
            while (i < word.items.len - 1) {
                if (word.items[i] == pair.first and word.items[i + 1] == pair.second) {
                    word.items[i] = pair_val.new_id;
                    _ = word.orderedRemove(i + 1);

                    // Merge offsets
                    char_offsets.items[i].end = char_offsets.items[i + 1].end;
                    _ = char_offsets.orderedRemove(i + 1);
                } else {
                    i += 1;
                }
            }
        }

        // Build result tokens
        var tokens = try allocator.alloc(Token, word.items.len);
        for (word.items, char_offsets.items, 0..) |id, off, i| {
            const token_str = self.vocab_r.get(id) orelse "";
            tokens[i] = Token.init(id, token_str, off.start, off.end);
        }

        return tokens;
    }

    /// Get vocabulary size
    pub fn getVocabSize(self: *const Self) usize {
        return self.vocab.count();
    }

    // ========================================================================
    // Fast Zero-Allocation Tokenization (O(n log n) algorithm)
    // ========================================================================

    /// Tokenize using linked-list + priority heap algorithm.
    /// Complexity: O(n log n) instead of O(k * n²)
    /// Zero allocations: uses pre-allocated arena buffers.
    ///
    /// Algorithm:
    /// 1. Initialize symbol linked list from input characters
    /// 2. Build priority heap of all adjacent pairs with merge ranks
    /// 3. Pop best (lowest rank) pair, merge symbols (O(1) with linked list)
    /// 4. Add new pairs created by merge to heap (O(log n))
    /// 5. Repeat until no more merges possible
    /// 6. Traverse linked list to collect final tokens
    pub fn tokenizeFast(self: *Self, arena: *TokenizerArena, sequence: []const u8) void {
        if (sequence.len == 0) return;

        const symbols = arena.bpe_symbols;
        var symbol_count: u16 = 0;
        const head: u16 = 0;

        // Step 1: Initialize symbol linked list from input characters
        var byte_idx: u32 = 0;
        var iter = std.unicode.Utf8Iterator{ .bytes = sequence, .i = 0 };

        while (iter.nextCodepointSlice()) |char_slice| {
            const char_len: u32 = @intCast(char_slice.len);
            const char_end = byte_idx + char_len;

            // Look up character in vocab
            const char_id = self.vocab.get(char_slice) orelse blk: {
                // Unknown character - use UNK token if available
                if (self.unk_token) |unk| {
                    break :blk self.vocab.get(unk) orelse {
                        byte_idx = char_end;
                        continue; // Skip if no UNK
                    };
                }
                byte_idx = char_end;
                continue; // Skip unknown char
            };

            // Add symbol to linked list (check bounds)
            const idx = symbol_count;
            if (idx >= symbols.len) {
                // Symbol buffer full, stop processing (truncate input)
                break;
            }
            symbols[idx] = .{
                .id = char_id,
                .start = byte_idx,
                .end = char_end,
                .prev = if (idx == 0) BPESymbol.SENTINEL else idx - 1,
                .next = BPESymbol.SENTINEL,
            };

            // Link previous symbol to this one
            if (idx > 0) {
                symbols[idx - 1].next = idx;
            }

            symbol_count += 1;
            byte_idx = char_end;
        }

        if (symbol_count == 0) return;
        if (symbol_count == 1) {
            // Single symbol - just add it (gracefully handle full buffer)
            _ = arena.encoding.tryAppend(SpanToken.init(
                symbols[0].id,
                symbols[0].start,
                symbols[0].end,
            ));
            return;
        }

        // Step 2: Build priority heap of all adjacent pairs
        arena.bpe_heap.reset();
        var idx: u16 = head;
        while (symbols[idx].next != BPESymbol.SENTINEL) {
            const next_idx = symbols[idx].next;
            const pair = Pair{ .first = symbols[idx].id, .second = symbols[next_idx].id };

            if (self.merges.get(pair.hash())) |pair_val| {
                arena.bpe_heap.insert(.{
                    .left_idx = idx,
                    .right_idx = next_idx,
                    .rank = pair_val.rank,
                });
            }
            idx = next_idx;
        }

        // Step 3-4: Process merges in priority order
        while (arena.bpe_heap.pop()) |best| {
            const left = &symbols[best.left_idx];
            const right = &symbols[best.right_idx];

            // Skip if either symbol was already merged (stale entry)
            if (left.next != best.right_idx) continue;
            if (right.isRemoved()) continue;

            // Get merge result
            const pair = Pair{ .first = left.id, .second = right.id };
            const pair_val = self.merges.get(pair.hash()) orelse continue;

            // Merge: update left symbol, remove right from list
            left.id = pair_val.new_id;
            left.end = right.end;
            left.next = right.next;

            // Update next symbol's prev pointer
            if (right.next != BPESymbol.SENTINEL) {
                symbols[right.next].prev = best.left_idx;
            }

            // Mark right as removed
            right.markRemoved();

            // Add new pairs to heap
            // New pair: prev <-> merged_left
            if (left.prev != BPESymbol.SENTINEL) {
                const prev_sym = &symbols[left.prev];
                const new_pair = Pair{ .first = prev_sym.id, .second = left.id };
                if (self.merges.get(new_pair.hash())) |m| {
                    arena.bpe_heap.insert(.{
                        .left_idx = left.prev,
                        .right_idx = best.left_idx,
                        .rank = m.rank,
                    });
                }
            }

            // New pair: merged_left <-> next
            if (left.next != BPESymbol.SENTINEL) {
                const next_sym = &symbols[left.next];
                const new_pair = Pair{ .first = left.id, .second = next_sym.id };
                if (self.merges.get(new_pair.hash())) |m| {
                    arena.bpe_heap.insert(.{
                        .left_idx = best.left_idx,
                        .right_idx = left.next,
                        .rank = m.rank,
                    });
                }
            }
        }

        // Step 5: Traverse linked list to collect final tokens
        idx = head;
        while (idx != BPESymbol.SENTINEL) {
            const sym = &symbols[idx];
            if (!sym.isRemoved()) {
                // Use tryAppend to gracefully handle buffer full (truncate long sequences)
                if (!arena.encoding.tryAppend(SpanToken.init(sym.id, sym.start, sym.end))) {
                    return; // Buffer full, stop collecting tokens
                }
            }
            idx = sym.next;
        }
    }

    /// Tokenize a word and append results to encoding (zero-alloc wrapper)
    pub fn tokenizeWord(self: *Self, arena: *TokenizerArena, word_start: u32, word_end: u32) void {
        const sequence = arena.encoding.input[word_start..word_end];
        if (sequence.len == 0) return;

        // Adjust offsets in the result
        const start_len = arena.encoding.len;
        self.tokenizeFast(arena, sequence);

        // Adjust offsets to be relative to original input
        var i: u32 = start_len;
        while (i < arena.encoding.len) : (i += 1) {
            arena.encoding.tokens[i].start += word_start;
            arena.encoding.tokens[i].end += word_start;
            arena.encoding.offsets[i].start += word_start;
            arena.encoding.offsets[i].end += word_start;
        }
    }
};

// ============================================================================
// Unit Tests
// ============================================================================

fn createTestBPEVocab(allocator: std.mem.Allocator) !struct {
    vocab: std.StringHashMapUnmanaged(u32),
    merges: std.AutoHashMapUnmanaged(u64, PairVal),
} {
    var vocab = std.StringHashMapUnmanaged(u32){};
    var merges = std.AutoHashMapUnmanaged(u64, PairVal){};

    // Build a simple BPE vocabulary
    // Individual characters first
    const chars = [_]struct { c: []const u8, id: u32 }{
        .{ .c = "h", .id = 0 },
        .{ .c = "e", .id = 1 },
        .{ .c = "l", .id = 2 },
        .{ .c = "o", .id = 3 },
        .{ .c = " ", .id = 4 },
        .{ .c = "w", .id = 5 },
        .{ .c = "r", .id = 6 },
        .{ .c = "d", .id = 7 },
        .{ .c = "<unk>", .id = 8 },
        // Merged tokens
        .{ .c = "he", .id = 9 },
        .{ .c = "ll", .id = 10 },
        .{ .c = "lo", .id = 11 },
        .{ .c = "hel", .id = 12 },
        .{ .c = "hell", .id = 13 },
        .{ .c = "hello", .id = 14 },
    };

    for (chars) |ch| {
        const key = try allocator.dupe(u8, ch.c);
        try vocab.put(allocator, key, ch.id);
    }

    // Merges (pair -> new_id, rank)
    // For "hello" to merge fully, we need: h+e->he, then he+ll->hell, then hell+o->hello
    // The rank determines priority - lower rank merges first
    // h + e -> he (rank 0) - first merge
    try merges.put(allocator, (Pair{ .first = 0, .second = 1 }).hash(), .{ .rank = 0, .new_id = 9 });
    // l + l -> ll (rank 1) - second merge
    try merges.put(allocator, (Pair{ .first = 2, .second = 2 }).hash(), .{ .rank = 1, .new_id = 10 });
    // he + ll -> hell (rank 2) - after he and ll exist, merge them
    try merges.put(allocator, (Pair{ .first = 9, .second = 10 }).hash(), .{ .rank = 2, .new_id = 13 });
    // hell + o -> hello (rank 3) - final merge
    try merges.put(allocator, (Pair{ .first = 13, .second = 3 }).hash(), .{ .rank = 3, .new_id = 14 });

    return .{ .vocab = vocab, .merges = merges };
}

test "bpe tokenize single word with merges" {
    const allocator = std.testing.allocator;
    const data = try createTestBPEVocab(allocator);
    const vocab = data.vocab;
    const merges = data.merges;

    var bpe = try BPE.init(allocator, vocab, merges, .{ .unk_token = "<unk>" });
    defer bpe.deinit();

    const tokens = try bpe.tokenize(allocator, "hello");
    defer allocator.free(tokens);

    // "hello" should be merged into a single token
    try std.testing.expectEqual(@as(usize, 1), tokens.len);
    try std.testing.expectEqual(@as(u32, 14), tokens[0].id);
}

test "bpe tokenize empty string" {
    const allocator = std.testing.allocator;
    const data = try createTestBPEVocab(allocator);
    const vocab = data.vocab;
    const merges = data.merges;

    var bpe = try BPE.init(allocator, vocab, merges, .{});
    defer bpe.deinit();

    const tokens = try bpe.tokenize(allocator, "");
    // Empty string returns empty slice (not allocated)

    try std.testing.expectEqual(@as(usize, 0), tokens.len);
}

test "bpe tokenize single char" {
    const allocator = std.testing.allocator;
    const data = try createTestBPEVocab(allocator);
    const vocab = data.vocab;
    const merges = data.merges;

    var bpe = try BPE.init(allocator, vocab, merges, .{});
    defer bpe.deinit();

    const tokens = try bpe.tokenize(allocator, "h");
    defer allocator.free(tokens);

    try std.testing.expectEqual(@as(usize, 1), tokens.len);
    try std.testing.expectEqual(@as(u32, 0), tokens[0].id);
}

test "bpe vocab size" {
    const allocator = std.testing.allocator;
    const data = try createTestBPEVocab(allocator);
    const vocab = data.vocab;
    const merges = data.merges;

    var bpe = try BPE.init(allocator, vocab, merges, .{});
    defer bpe.deinit();

    try std.testing.expectEqual(@as(usize, 15), bpe.getVocabSize());
}

test "bpe model interface" {
    const allocator = std.testing.allocator;
    const data = try createTestBPEVocab(allocator);
    const vocab = data.vocab;
    const merges = data.merges;

    var bpe = try BPE.init(allocator, vocab, merges, .{});
    defer bpe.deinit();

    const m = bpe.getModel();

    // tokenToId
    try std.testing.expectEqual(@as(u32, 14), m.tokenToId("hello").?);
    try std.testing.expectEqual(@as(u32, 0), m.tokenToId("h").?);
    try std.testing.expect(m.tokenToId("notfound") == null);

    // idToToken
    try std.testing.expectEqualStrings("hello", m.idToToken(14).?);
    try std.testing.expectEqualStrings("h", m.idToToken(0).?);
    try std.testing.expect(m.idToToken(99999) == null);

    // getVocabSize
    try std.testing.expectEqual(@as(usize, 15), m.getVocabSize());
}

test "bpe pair hash" {
    const pair1 = Pair{ .first = 0, .second = 1 };
    const pair2 = Pair{ .first = 0, .second = 1 };
    const pair3 = Pair{ .first = 1, .second = 0 };

    try std.testing.expectEqual(pair1.hash(), pair2.hash());
    try std.testing.expect(pair1.hash() != pair3.hash());
}

test "bpe token offsets preserved" {
    const allocator = std.testing.allocator;
    const data = try createTestBPEVocab(allocator);
    const vocab = data.vocab;
    const merges = data.merges;

    var bpe = try BPE.init(allocator, vocab, merges, .{});
    defer bpe.deinit();

    const tokens = try bpe.tokenize(allocator, "hello");
    defer allocator.free(tokens);

    // "hello" merged to single token spanning full word
    try std.testing.expectEqual(@as(u32, 0), tokens[0].offset.start);
    try std.testing.expectEqual(@as(u32, 5), tokens[0].offset.end);
}

test "bpe no merges possible" {
    const allocator = std.testing.allocator;

    // Create vocab with chars but no merges
    var vocab = std.StringHashMapUnmanaged(u32){};
    try vocab.put(allocator, try allocator.dupe(u8, "a"), 0);
    try vocab.put(allocator, try allocator.dupe(u8, "b"), 1);
    try vocab.put(allocator, try allocator.dupe(u8, "c"), 2);

    // Empty merge rules
    const merges = std.AutoHashMapUnmanaged(u64, PairVal){};

    var bpe = try BPE.init(allocator, vocab, merges, .{});
    defer bpe.deinit();

    // Without merges, each char stays separate
    const tokens = try bpe.tokenize(allocator, "abc");
    defer allocator.free(tokens);

    try std.testing.expectEqual(@as(usize, 3), tokens.len);
    try std.testing.expectEqual(@as(u32, 0), tokens[0].id); // "a"
    try std.testing.expectEqual(@as(u32, 1), tokens[1].id); // "b"
    try std.testing.expectEqual(@as(u32, 2), tokens[2].id); // "c"
}

test "bpe char not in vocab skipped" {
    const allocator = std.testing.allocator;
    const data = try createTestBPEVocab(allocator);
    const vocab = data.vocab;
    const merges = data.merges;

    var bpe = try BPE.init(allocator, vocab, merges, .{});
    defer bpe.deinit();

    // 'z' is not in vocab, should be skipped (no unk_token set)
    const tokens = try bpe.tokenize(allocator, "hz");
    defer allocator.free(tokens);

    try std.testing.expectEqual(@as(usize, 1), tokens.len);
    try std.testing.expectEqual(@as(u32, 0), tokens[0].id); // "h"
}

test "bpe two char word no merge" {
    const allocator = std.testing.allocator;

    var vocab = std.StringHashMapUnmanaged(u32){};
    try vocab.put(allocator, try allocator.dupe(u8, "x"), 0);
    try vocab.put(allocator, try allocator.dupe(u8, "y"), 1);

    const merges = std.AutoHashMapUnmanaged(u64, PairVal){};

    var bpe = try BPE.init(allocator, vocab, merges, .{});
    defer bpe.deinit();

    const tokens = try bpe.tokenize(allocator, "xy");
    defer allocator.free(tokens);

    // No merge rule for "xy", so stays as two tokens
    try std.testing.expectEqual(@as(usize, 2), tokens.len);
}

test "bpe multiple merges chain" {
    const allocator = std.testing.allocator;
    const data = try createTestBPEVocab(allocator);
    const vocab = data.vocab;
    const merges = data.merges;

    var bpe = try BPE.init(allocator, vocab, merges, .{});
    defer bpe.deinit();

    // "hello" should go through multiple merges:
    // h,e,l,l,o -> he,l,l,o -> hel,l,o -> hell,o -> hello
    const tokens = try bpe.tokenize(allocator, "hello");
    defer allocator.free(tokens);

    try std.testing.expectEqual(@as(usize, 1), tokens.len);
    try std.testing.expectEqualStrings("hello", tokens[0].value);
}

test "bpe pair hash equality" {
    const pair1 = Pair{ .first = 10, .second = 20 };
    const pair2 = Pair{ .first = 10, .second = 20 };
    const pair3 = Pair{ .first = 20, .second = 10 };

    // Same pairs have same hash
    try std.testing.expectEqual(pair1.hash(), pair2.hash());
    // Different pairs have different hash
    try std.testing.expect(pair1.hash() != pair3.hash());
}

// ============================================================================
// Fast BPE (Zero-Allocation) Unit Tests
// ============================================================================

test "bpe tokenizeFast single word with merges" {
    const allocator = std.testing.allocator;
    const data = try createTestBPEVocab(allocator);
    const vocab = data.vocab;
    const merges = data.merges;

    var bpe = try BPE.init(allocator, vocab, merges, .{ .unk_token = "<unk>" });
    defer bpe.deinit();

    var arena = try TokenizerArena.init(allocator, .{});
    defer arena.deinit();

    arena.reset("hello");
    bpe.tokenizeFast(&arena, "hello");

    // "hello" should be merged into a single token
    try std.testing.expectEqual(@as(u32, 1), arena.encoding.len);
    try std.testing.expectEqual(@as(u32, 14), arena.encoding.getIds()[0]);
}

test "bpe tokenizeFast empty string" {
    const allocator = std.testing.allocator;
    const data = try createTestBPEVocab(allocator);
    const vocab = data.vocab;
    const merges = data.merges;

    var bpe = try BPE.init(allocator, vocab, merges, .{});
    defer bpe.deinit();

    var arena = try TokenizerArena.init(allocator, .{});
    defer arena.deinit();

    arena.reset("");
    bpe.tokenizeFast(&arena, "");

    try std.testing.expectEqual(@as(u32, 0), arena.encoding.len);
}

test "bpe tokenizeFast single char" {
    const allocator = std.testing.allocator;
    const data = try createTestBPEVocab(allocator);
    const vocab = data.vocab;
    const merges = data.merges;

    var bpe = try BPE.init(allocator, vocab, merges, .{});
    defer bpe.deinit();

    var arena = try TokenizerArena.init(allocator, .{});
    defer arena.deinit();

    arena.reset("h");
    bpe.tokenizeFast(&arena, "h");

    try std.testing.expectEqual(@as(u32, 1), arena.encoding.len);
    try std.testing.expectEqual(@as(u32, 0), arena.encoding.getIds()[0]);
}

test "bpe tokenizeFast offsets preserved" {
    const allocator = std.testing.allocator;
    const data = try createTestBPEVocab(allocator);
    const vocab = data.vocab;
    const merges = data.merges;

    var bpe = try BPE.init(allocator, vocab, merges, .{});
    defer bpe.deinit();

    var arena = try TokenizerArena.init(allocator, .{});
    defer arena.deinit();

    arena.reset("hello");
    bpe.tokenizeFast(&arena, "hello");

    // "hello" merged to single token spanning full word
    const offsets = arena.encoding.getOffsets();
    try std.testing.expectEqual(@as(u32, 0), offsets[0].start);
    try std.testing.expectEqual(@as(u32, 5), offsets[0].end);
}

test "bpe tokenizeFast no merges" {
    const allocator = std.testing.allocator;

    // Create vocab with chars but no merges
    var vocab = std.StringHashMapUnmanaged(u32){};
    try vocab.put(allocator, try allocator.dupe(u8, "a"), 0);
    try vocab.put(allocator, try allocator.dupe(u8, "b"), 1);
    try vocab.put(allocator, try allocator.dupe(u8, "c"), 2);

    const merges = std.AutoHashMapUnmanaged(u64, PairVal){};

    var bpe = try BPE.init(allocator, vocab, merges, .{});
    defer bpe.deinit();

    var arena = try TokenizerArena.init(allocator, .{});
    defer arena.deinit();

    arena.reset("abc");
    bpe.tokenizeFast(&arena, "abc");

    // Without merges, each char stays separate
    try std.testing.expectEqual(@as(u32, 3), arena.encoding.len);
    const ids = arena.encoding.getIds();
    try std.testing.expectEqual(@as(u32, 0), ids[0]); // "a"
    try std.testing.expectEqual(@as(u32, 1), ids[1]); // "b"
    try std.testing.expectEqual(@as(u32, 2), ids[2]); // "c"
}

test "bpe tokenizeFast matches original" {
    const allocator = std.testing.allocator;
    const data = try createTestBPEVocab(allocator);
    const vocab = data.vocab;
    const merges = data.merges;

    var bpe = try BPE.init(allocator, vocab, merges, .{ .unk_token = "<unk>" });
    defer bpe.deinit();

    // Test with original algorithm
    const original_tokens = try bpe.tokenize(allocator, "hello");
    defer allocator.free(original_tokens);

    // Test with fast algorithm
    var arena = try TokenizerArena.init(allocator, .{});
    defer arena.deinit();

    arena.reset("hello");
    bpe.tokenizeFast(&arena, "hello");

    // Should produce same results
    try std.testing.expectEqual(original_tokens.len, arena.encoding.len);
    for (original_tokens, 0..) |tok, i| {
        try std.testing.expectEqual(tok.id, arena.encoding.getIds()[i]);
        try std.testing.expectEqual(tok.offset.start, arena.encoding.getOffsets()[i].start);
        try std.testing.expectEqual(tok.offset.end, arena.encoding.getOffsets()[i].end);
    }
}

test "bpe tokenizeFast arena reuse" {
    const allocator = std.testing.allocator;
    const data = try createTestBPEVocab(allocator);
    const vocab = data.vocab;
    const merges = data.merges;

    var bpe = try BPE.init(allocator, vocab, merges, .{});
    defer bpe.deinit();

    var arena = try TokenizerArena.init(allocator, .{});
    defer arena.deinit();

    // First tokenization
    arena.reset("hello");
    bpe.tokenizeFast(&arena, "hello");
    try std.testing.expectEqual(@as(u32, 1), arena.encoding.len);

    // Reset and reuse - no allocation
    arena.reset("he");
    bpe.tokenizeFast(&arena, "he");
    try std.testing.expectEqual(@as(u32, 1), arena.encoding.len);
    try std.testing.expectEqual(@as(u32, 9), arena.encoding.getIds()[0]); // "he" merged
}
