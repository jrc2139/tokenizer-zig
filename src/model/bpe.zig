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

    const Self = @This();

    pub const Config = struct {
        unk_token: ?[]const u8 = null,
        continuing_subword_prefix: ?[]const u8 = null,
        end_of_word_suffix: ?[]const u8 = null,
        dropout: ?f32 = null,
    };

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
        };
    }

    pub fn deinit(self: *Self) void {
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

            if (self.vocab.get(char_str)) |id| {
                try word.append(allocator, id);
            } else if (self.unk_token) |unk| {
                if (self.vocab.get(unk)) |unk_id| {
                    try word.append(allocator, unk_id);
                }
            }

            const char_len: u32 = @intCast(char_slice.len);
            try char_offsets.append(allocator, .{
                .start = byte_idx,
                .end = byte_idx + char_len,
            });
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
};
