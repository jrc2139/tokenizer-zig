//! WordPiece tokenization model
//!
//! WordPiece is used by BERT and similar models. It splits words into
//! subword units, with non-first subwords prefixed with "##".

const std = @import("std");
const model = @import("model.zig");
const Token = @import("../token.zig").Token;
const SpanToken = @import("../token.zig").SpanToken;
const TokenizerArena = @import("../arena.zig").TokenizerArena;

/// WordPiece tokenization model
pub const WordPiece = struct {
    allocator: std.mem.Allocator,
    vocab: std.StringHashMapUnmanaged(u32),
    vocab_r: std.AutoHashMapUnmanaged(u32, []const u8),
    unk_token: []const u8,
    continuing_subword_prefix: []const u8,
    max_input_chars_per_word: usize,
    owns_strings: bool, // Whether we own unk_token and continuing_subword_prefix

    const Self = @This();

    pub const Config = struct {
        unk_token: []const u8 = "[UNK]",
        continuing_subword_prefix: []const u8 = "##",
        max_input_chars_per_word: usize = 100,
    };

    /// Initialize with borrowed strings (caller retains ownership)
    pub fn init(allocator: std.mem.Allocator, vocab: std.StringHashMapUnmanaged(u32), config: Config) !Self {
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
            .unk_token = config.unk_token,
            .continuing_subword_prefix = config.continuing_subword_prefix,
            .max_input_chars_per_word = config.max_input_chars_per_word,
            .owns_strings = false,
        };
    }

    /// Initialize with owned strings (WordPiece takes ownership and frees on deinit)
    pub fn initOwned(
        allocator: std.mem.Allocator,
        vocab: std.StringHashMapUnmanaged(u32),
        unk_token: []const u8,
        prefix: []const u8,
        max_chars: usize,
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
            .unk_token = unk_token,
            .continuing_subword_prefix = prefix,
            .max_input_chars_per_word = max_chars,
            .owns_strings = true,
        };
    }

    pub fn deinit(self: *Self) void {
        // Free owned config strings
        if (self.owns_strings) {
            self.allocator.free(self.unk_token);
            self.allocator.free(self.continuing_subword_prefix);
        }

        // Free vocab keys (strings)
        var it = self.vocab.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.vocab.deinit(self.allocator);
        self.vocab_r.deinit(self.allocator);
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

    /// Get vocabulary size
    pub fn getVocabSize(self: *const Self) usize {
        return self.vocab.count();
    }

    /// Tokenize a single word using WordPiece algorithm
    pub fn tokenize(self: *Self, allocator: std.mem.Allocator, sequence: []const u8) ![]Token {
        var tokens = std.ArrayListUnmanaged(Token){};
        errdefer tokens.deinit(allocator);

        const chars = sequence;
        const char_len = chars.len;

        // If word is too long, return UNK
        if (char_len > self.max_input_chars_per_word) {
            const unk_id = self.vocab.get(self.unk_token) orelse return error.MissingUnkToken;
            try tokens.append(allocator, Token.init(
                unk_id,
                self.unk_token,
                0,
                @intCast(char_len),
            ));
            return try tokens.toOwnedSlice(allocator);
        }

        var is_bad = false;
        var start: usize = 0;

        while (start < char_len) {
            var end = char_len;
            var cur_substr: ?[]const u8 = null;
            var cur_id: ?u32 = null;

            while (start < end) {
                var substr_buf: [512]u8 = undefined;
                var substr: []const u8 = undefined;

                if (start > 0) {
                    // Add continuing subword prefix
                    const prefix_len = self.continuing_subword_prefix.len;
                    const word_len = end - start;
                    if (prefix_len + word_len > substr_buf.len) {
                        end -= 1;
                        continue;
                    }
                    @memcpy(substr_buf[0..prefix_len], self.continuing_subword_prefix);
                    @memcpy(substr_buf[prefix_len .. prefix_len + word_len], chars[start..end]);
                    substr = substr_buf[0 .. prefix_len + word_len];
                } else {
                    substr = chars[start..end];
                }

                if (self.vocab.get(substr)) |id| {
                    cur_substr = substr;
                    cur_id = id;
                    break;
                }
                end -= 1;
            }

            if (cur_substr == null) {
                is_bad = true;
                break;
            }

            try tokens.append(allocator, Token.init(
                cur_id.?,
                cur_substr.?,
                @intCast(start),
                @intCast(end),
            ));
            start = end;
        }

        if (is_bad) {
            // Clear tokens and return UNK
            tokens.clearAndFree(allocator);
            const unk_id = self.vocab.get(self.unk_token) orelse return error.MissingUnkToken;
            try tokens.append(allocator, Token.init(
                unk_id,
                self.unk_token,
                0,
                @intCast(char_len),
            ));
        }

        return try tokens.toOwnedSlice(allocator);
    }

    /// Zero-allocation tokenization using arena buffers.
    /// Writes SpanTokens directly to arena.encoding.
    ///
    /// Algorithm: Greedy longest-match
    /// 1. Start from beginning of word
    /// 2. Find longest substring in vocabulary
    /// 3. For non-first tokens, prepend continuing_subword_prefix
    /// 4. Repeat until end of word
    /// 5. If any position fails, output single UNK token
    pub fn tokenizeFast(self: *Self, arena: *TokenizerArena, sequence: []const u8) void {
        if (sequence.len == 0) return;

        const chars = sequence;
        const char_len = chars.len;

        // If word is too long, return UNK
        if (char_len > self.max_input_chars_per_word) {
            const unk_id = self.vocab.get(self.unk_token) orelse return;
            arena.encoding.append(SpanToken.init(unk_id, 0, @intCast(char_len)));
            return;
        }

        // Track tokens we've found (we may need to discard if we hit a bad segment)
        const start_len = arena.encoding.len;
        var is_bad = false;
        var start: u32 = 0;

        while (start < char_len) {
            var end: u32 = @intCast(char_len);
            var cur_id: ?u32 = null;
            var matched_end: u32 = 0;

            while (start < end) {
                var substr_buf: [512]u8 = undefined;
                var substr: []const u8 = undefined;

                if (start > 0) {
                    // Add continuing subword prefix
                    const prefix_len = self.continuing_subword_prefix.len;
                    const word_len = end - start;
                    if (prefix_len + word_len > substr_buf.len) {
                        end -= 1;
                        continue;
                    }
                    @memcpy(substr_buf[0..prefix_len], self.continuing_subword_prefix);
                    @memcpy(substr_buf[prefix_len .. prefix_len + word_len], chars[start..end]);
                    substr = substr_buf[0 .. prefix_len + word_len];
                } else {
                    substr = chars[start..end];
                }

                if (self.vocab.get(substr)) |id| {
                    cur_id = id;
                    matched_end = end;
                    break;
                }
                end -= 1;
            }

            if (cur_id == null) {
                is_bad = true;
                break;
            }

            // Append token with correct offsets
            arena.encoding.append(SpanToken.init(cur_id.?, start, matched_end));
            start = matched_end;
        }

        if (is_bad) {
            // Rollback any tokens we added and output single UNK
            arena.encoding.len = start_len;
            const unk_id = self.vocab.get(self.unk_token) orelse return;
            arena.encoding.append(SpanToken.init(unk_id, 0, @intCast(char_len)));
        }
    }
};

// ============================================================================
// Unit Tests
// ============================================================================

fn createTestVocab(allocator: std.mem.Allocator) !std.StringHashMapUnmanaged(u32) {
    var vocab = std.StringHashMapUnmanaged(u32){};

    // Basic BERT-like vocabulary
    const words = [_]struct { word: []const u8, id: u32 }{
        .{ .word = "[UNK]", .id = 0 },
        .{ .word = "[CLS]", .id = 101 },
        .{ .word = "[SEP]", .id = 102 },
        .{ .word = "hello", .id = 7592 },
        .{ .word = "world", .id = 2088 },
        .{ .word = "un", .id = 4895 },
        .{ .word = "##known", .id = 5765 },
        .{ .word = "play", .id = 2377 },
        .{ .word = "##ing", .id = 2075 },
        .{ .word = "##s", .id = 1055 },
        .{ .word = "the", .id = 1996 },
        .{ .word = "a", .id = 1037 },
        .{ .word = "cat", .id = 4937 },
        .{ .word = "dog", .id = 3899 },
    };

    for (words) |w| {
        const key = try allocator.dupe(u8, w.word);
        try vocab.put(allocator, key, w.id);
    }

    return vocab;
}

test "wordpiece tokenize single known word" {
    const allocator = std.testing.allocator;
    const vocab = try createTestVocab(allocator);
    var wp = try WordPiece.init(allocator, vocab, .{});
    defer wp.deinit();

    const tokens = try wp.tokenize(allocator, "hello");
    defer allocator.free(tokens);

    try std.testing.expectEqual(@as(usize, 1), tokens.len);
    try std.testing.expectEqual(@as(u32, 7592), tokens[0].id);
    try std.testing.expectEqualStrings("hello", tokens[0].value);
}

test "wordpiece tokenize with subword split" {
    const allocator = std.testing.allocator;
    const vocab = try createTestVocab(allocator);
    var wp = try WordPiece.init(allocator, vocab, .{});
    defer wp.deinit();

    // "unknown" should split into "un" + "##known"
    const tokens = try wp.tokenize(allocator, "unknown");
    defer allocator.free(tokens);

    try std.testing.expectEqual(@as(usize, 2), tokens.len);
    try std.testing.expectEqual(@as(u32, 4895), tokens[0].id); // "un"
    try std.testing.expectEqual(@as(u32, 5765), tokens[1].id); // "##known"
}

test "wordpiece tokenize with suffix" {
    const allocator = std.testing.allocator;
    const vocab = try createTestVocab(allocator);
    var wp = try WordPiece.init(allocator, vocab, .{});
    defer wp.deinit();

    // "playing" should split into "play" + "##ing"
    const tokens = try wp.tokenize(allocator, "playing");
    defer allocator.free(tokens);

    try std.testing.expectEqual(@as(usize, 2), tokens.len);
    try std.testing.expectEqual(@as(u32, 2377), tokens[0].id); // "play"
    try std.testing.expectEqual(@as(u32, 2075), tokens[1].id); // "##ing"
}

test "wordpiece tokenize unknown word returns UNK" {
    const allocator = std.testing.allocator;
    const vocab = try createTestVocab(allocator);
    var wp = try WordPiece.init(allocator, vocab, .{});
    defer wp.deinit();

    // "xyz" is not in vocab and can't be split
    const tokens = try wp.tokenize(allocator, "xyz");
    defer allocator.free(tokens);

    try std.testing.expectEqual(@as(usize, 1), tokens.len);
    try std.testing.expectEqual(@as(u32, 0), tokens[0].id); // [UNK]
}

test "wordpiece tokenize empty string" {
    const allocator = std.testing.allocator;
    const vocab = try createTestVocab(allocator);
    var wp = try WordPiece.init(allocator, vocab, .{});
    defer wp.deinit();

    const tokens = try wp.tokenize(allocator, "");
    defer allocator.free(tokens);

    try std.testing.expectEqual(@as(usize, 0), tokens.len);
}

test "wordpiece tokenize word too long" {
    const allocator = std.testing.allocator;
    const vocab = try createTestVocab(allocator);
    var wp = try WordPiece.init(allocator, vocab, .{ .max_input_chars_per_word = 5 });
    defer wp.deinit();

    // "helloworld" is 10 chars, max is 5
    const tokens = try wp.tokenize(allocator, "helloworld");
    defer allocator.free(tokens);

    try std.testing.expectEqual(@as(usize, 1), tokens.len);
    try std.testing.expectEqual(@as(u32, 0), tokens[0].id); // [UNK]
}

test "wordpiece vocab size" {
    const allocator = std.testing.allocator;
    const vocab = try createTestVocab(allocator);
    var wp = try WordPiece.init(allocator, vocab, .{});
    defer wp.deinit();

    try std.testing.expectEqual(@as(usize, 14), wp.getVocabSize());
}

test "wordpiece model interface" {
    const allocator = std.testing.allocator;
    const vocab = try createTestVocab(allocator);
    var wp = try WordPiece.init(allocator, vocab, .{});
    defer wp.deinit();

    const m = wp.getModel();

    // tokenToId
    try std.testing.expectEqual(@as(u32, 7592), m.tokenToId("hello").?);
    try std.testing.expect(m.tokenToId("notfound") == null);

    // idToToken
    try std.testing.expectEqualStrings("hello", m.idToToken(7592).?);
    try std.testing.expect(m.idToToken(99999) == null);

    // getVocabSize
    try std.testing.expectEqual(@as(usize, 14), m.getVocabSize());
}

test "wordpiece token offsets" {
    const allocator = std.testing.allocator;
    const vocab = try createTestVocab(allocator);
    var wp = try WordPiece.init(allocator, vocab, .{});
    defer wp.deinit();

    const tokens = try wp.tokenize(allocator, "playing");
    defer allocator.free(tokens);

    // "play" = 0..4, "##ing" = 4..7
    try std.testing.expectEqual(@as(u32, 0), tokens[0].offset.start);
    try std.testing.expectEqual(@as(u32, 4), tokens[0].offset.end);
    try std.testing.expectEqual(@as(u32, 4), tokens[1].offset.start);
    try std.testing.expectEqual(@as(u32, 7), tokens[1].offset.end);
}

test "wordpiece exactly at max_input_chars" {
    const allocator = std.testing.allocator;
    const vocab = try createTestVocab(allocator);
    var wp = try WordPiece.init(allocator, vocab, .{ .max_input_chars_per_word = 5 });
    defer wp.deinit();

    // "hello" is exactly 5 chars, should tokenize normally
    const tokens = try wp.tokenize(allocator, "hello");
    defer allocator.free(tokens);

    try std.testing.expectEqual(@as(usize, 1), tokens.len);
    try std.testing.expectEqual(@as(u32, 7592), tokens[0].id); // "hello"
}

test "wordpiece one over max_input_chars" {
    const allocator = std.testing.allocator;
    const vocab = try createTestVocab(allocator);
    var wp = try WordPiece.init(allocator, vocab, .{ .max_input_chars_per_word = 5 });
    defer wp.deinit();

    // "helloo" is 6 chars, max is 5, should return UNK
    const tokens = try wp.tokenize(allocator, "helloo");
    defer allocator.free(tokens);

    try std.testing.expectEqual(@as(usize, 1), tokens.len);
    try std.testing.expectEqual(@as(u32, 0), tokens[0].id); // [UNK]
}

test "wordpiece custom unk token" {
    const allocator = std.testing.allocator;

    // Create vocab with custom UNK - use the base vocab and check UNK behavior
    const vocab = try createTestVocab(allocator);
    var wp = try WordPiece.init(allocator, vocab, .{ .unk_token = "[UNK]" });
    defer wp.deinit();

    // Unknown word should return UNK token (ID 0 in createTestVocab)
    const tokens = try wp.tokenize(allocator, "xyz");
    defer allocator.free(tokens);

    try std.testing.expectEqual(@as(usize, 1), tokens.len);
    try std.testing.expectEqual(@as(u32, 0), tokens[0].id); // [UNK]
}

test "wordpiece continuing_subword_prefix default" {
    const allocator = std.testing.allocator;

    // Test that the default ## prefix works
    const vocab = try createTestVocab(allocator);
    var wp = try WordPiece.init(allocator, vocab, .{});
    defer wp.deinit();

    const tokens = try wp.tokenize(allocator, "playing");
    defer allocator.free(tokens);

    try std.testing.expectEqual(@as(usize, 2), tokens.len);
    try std.testing.expectEqualStrings("play", tokens[0].value);
    try std.testing.expectEqualStrings("##ing", tokens[1].value);
}

test "wordpiece single char word in vocab" {
    const allocator = std.testing.allocator;
    const vocab = try createTestVocab(allocator);
    var wp = try WordPiece.init(allocator, vocab, .{});
    defer wp.deinit();

    // Single char "a" IS in vocab (ID 1037)
    const tokens = try wp.tokenize(allocator, "a");
    defer allocator.free(tokens);

    try std.testing.expectEqual(@as(usize, 1), tokens.len);
    try std.testing.expectEqual(@as(u32, 1037), tokens[0].id); // "a"
}

test "wordpiece single char not in vocab" {
    const allocator = std.testing.allocator;
    const vocab = try createTestVocab(allocator);
    var wp = try WordPiece.init(allocator, vocab, .{});
    defer wp.deinit();

    // Single char "z" is NOT in vocab, returns UNK
    const tokens = try wp.tokenize(allocator, "z");
    defer allocator.free(tokens);

    try std.testing.expectEqual(@as(usize, 1), tokens.len);
    try std.testing.expectEqual(@as(u32, 0), tokens[0].id); // [UNK]
}

test "wordpiece multiple subword splits" {
    const allocator = std.testing.allocator;
    const vocab = try createTestVocab(allocator);
    var wp = try WordPiece.init(allocator, vocab, .{});
    defer wp.deinit();

    // "playing" -> "play" + "##ing"
    const tokens = try wp.tokenize(allocator, "playing");
    defer allocator.free(tokens);

    try std.testing.expectEqual(@as(usize, 2), tokens.len);
    try std.testing.expectEqualStrings("play", tokens[0].value);
    try std.testing.expectEqualStrings("##ing", tokens[1].value);
}
