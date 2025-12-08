//! BPE (Byte Pair Encoding) tokenization model
//!
//! BPE is used by GPT-2, RoBERTa, and many other models.

const std = @import("std");
const model = @import("model.zig");
const Token = @import("../token.zig").Token;

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
