//! PreTokenizer interface for pre-tokenization
//!
//! Pre-tokenizers split text before the actual tokenization model runs.
//! They handle word splitting, whitespace handling, etc.

const std = @import("std");

/// Pre-token with offset information
pub const PreToken = struct {
    content: []const u8,
    start: u32,
    end: u32,
};

/// Function pointer types
pub const PreTokenizeFn = *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, input: []const u8) anyerror![]const []const u8;
pub const DeinitFn = *const fn (ctx: *anyopaque) void;

/// PreTokenizer interface
pub const PreTokenizer = struct {
    ptr: *anyopaque,
    pre_tokenize_fn: PreTokenizeFn,
    deinit_fn: ?DeinitFn = null,

    const Self = @This();

    pub fn preTokenize(self: Self, allocator: std.mem.Allocator, input: []const u8) ![]const []const u8 {
        return self.pre_tokenize_fn(self.ptr, allocator, input);
    }

    pub fn deinit(self: *Self) void {
        if (self.deinit_fn) |f| {
            f(self.ptr);
        }
    }
};

/// Whitespace pre-tokenizer - splits on whitespace
pub const Whitespace = struct {
    const Self = @This();

    pub fn preTokenizer(self: *Self) PreTokenizer {
        return .{
            .ptr = self,
            .pre_tokenize_fn = preTokenizeImpl,
        };
    }

    fn preTokenizeImpl(_: *anyopaque, allocator: std.mem.Allocator, input: []const u8) anyerror![]const []const u8 {
        var tokens = std.ArrayListUnmanaged([]const u8){};
        errdefer tokens.deinit(allocator);

        var start: usize = 0;
        var in_word = false;

        for (input, 0..) |c, i| {
            const is_ws = c == ' ' or c == '\t' or c == '\n' or c == '\r';
            if (is_ws) {
                if (in_word) {
                    try tokens.append(allocator, input[start..i]);
                    in_word = false;
                }
            } else {
                if (!in_word) {
                    start = i;
                    in_word = true;
                }
            }
        }

        // Handle last word
        if (in_word) {
            try tokens.append(allocator, input[start..]);
        }

        return try tokens.toOwnedSlice(allocator);
    }
};

/// BERT pre-tokenizer - whitespace splitting with special handling
pub const BertPreTokenizer = struct {
    const Self = @This();

    pub fn preTokenizer(self: *Self) PreTokenizer {
        return .{
            .ptr = self,
            .pre_tokenize_fn = preTokenizeImpl,
        };
    }

    fn preTokenizeImpl(_: *anyopaque, allocator: std.mem.Allocator, input: []const u8) anyerror![]const []const u8 {
        var tokens = std.ArrayListUnmanaged([]const u8){};
        errdefer tokens.deinit(allocator);

        var start: usize = 0;
        var in_word = false;

        for (input, 0..) |c, i| {
            const is_ws = c == ' ' or c == '\t' or c == '\n' or c == '\r';
            const is_punct = isPunctuation(c);

            if (is_ws or is_punct) {
                if (in_word) {
                    try tokens.append(allocator, input[start..i]);
                    in_word = false;
                }
                if (is_punct) {
                    // Punctuation is its own token
                    try tokens.append(allocator, input[i .. i + 1]);
                }
            } else {
                if (!in_word) {
                    start = i;
                    in_word = true;
                }
            }
        }

        // Handle last word
        if (in_word) {
            try tokens.append(allocator, input[start..]);
        }

        return try tokens.toOwnedSlice(allocator);
    }

    fn isPunctuation(c: u8) bool {
        return (c >= 33 and c <= 47) or // !"#$%&'()*+,-./
            (c >= 58 and c <= 64) or // :;<=>?@
            (c >= 91 and c <= 96) or // [\]^_`
            (c >= 123 and c <= 126); // {|}~
    }
};

/// ByteLevel pre-tokenizer (used by GPT-2)
pub const ByteLevel = struct {
    add_prefix_space: bool = true,
    trim_offsets: bool = true,
    use_regex: bool = true,

    const Self = @This();

    pub fn preTokenizer(self: *Self) PreTokenizer {
        return .{
            .ptr = self,
            .pre_tokenize_fn = preTokenizeImpl,
        };
    }

    fn preTokenizeImpl(ctx: *anyopaque, allocator: std.mem.Allocator, input: []const u8) anyerror![]const []const u8 {
        const self: *Self = @ptrCast(@alignCast(ctx));
        _ = self;

        // Simple implementation: just split on whitespace for now
        // Full implementation would use the GPT-2 regex pattern
        var tokens = std.ArrayListUnmanaged([]const u8){};
        errdefer tokens.deinit(allocator);

        var start: usize = 0;
        var in_word = false;

        for (input, 0..) |c, i| {
            const is_ws = c == ' ' or c == '\t' or c == '\n' or c == '\r';
            if (is_ws) {
                if (in_word) {
                    try tokens.append(allocator, input[start..i]);
                    in_word = false;
                }
            } else {
                if (!in_word) {
                    start = i;
                    in_word = true;
                }
            }
        }

        if (in_word) {
            try tokens.append(allocator, input[start..]);
        }

        return try tokens.toOwnedSlice(allocator);
    }
};

/// Sequence pre-tokenizer - applies multiple pre-tokenizers
pub const Sequence = struct {
    pre_tokenizers: []PreTokenizer,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, pre_tokenizers: []const PreTokenizer) !Self {
        const owned = try allocator.dupe(PreTokenizer, pre_tokenizers);
        return .{
            .pre_tokenizers = owned,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.pre_tokenizers);
    }

    pub fn preTokenizer(self: *Self) PreTokenizer {
        return .{
            .ptr = self,
            .pre_tokenize_fn = preTokenizeImpl,
            .deinit_fn = deinitImpl,
        };
    }

    fn preTokenizeImpl(ctx: *anyopaque, allocator: std.mem.Allocator, input: []const u8) anyerror![]const []const u8 {
        const self: *Self = @ptrCast(@alignCast(ctx));

        var current: []const []const u8 = &.{input};
        var owned = false;

        for (self.pre_tokenizers) |pt| {
            var next_tokens = std.ArrayListUnmanaged([]const u8){};
            errdefer next_tokens.deinit(allocator);

            for (current) |token| {
                const sub_tokens = try pt.preTokenize(allocator, token);
                defer allocator.free(sub_tokens);
                try next_tokens.appendSlice(allocator, sub_tokens);
            }

            if (owned) {
                allocator.free(current);
            }
            current = try next_tokens.toOwnedSlice(allocator);
            owned = true;
        }

        if (!owned) {
            var result = try allocator.alloc([]const u8, 1);
            result[0] = input;
            return result;
        }
        return current;
    }

    fn deinitImpl(ctx: *anyopaque) void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        self.deinit();
    }
};

// ============================================================================
// Tests
// ============================================================================

test "Whitespace: splits on space" {
    const allocator = std.testing.allocator;
    var ws = Whitespace{};
    const pt = ws.preTokenizer();

    const result = try pt.preTokenize(allocator, "hello world");
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 2), result.len);
    try std.testing.expectEqualStrings("hello", result[0]);
    try std.testing.expectEqualStrings("world", result[1]);
}

test "Whitespace: splits on tab" {
    const allocator = std.testing.allocator;
    var ws = Whitespace{};
    const pt = ws.preTokenizer();

    const result = try pt.preTokenize(allocator, "hello\tworld");
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 2), result.len);
    try std.testing.expectEqualStrings("hello", result[0]);
    try std.testing.expectEqualStrings("world", result[1]);
}

test "Whitespace: splits on newline" {
    const allocator = std.testing.allocator;
    var ws = Whitespace{};
    const pt = ws.preTokenizer();

    const result = try pt.preTokenize(allocator, "hello\nworld");
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 2), result.len);
    try std.testing.expectEqualStrings("hello", result[0]);
    try std.testing.expectEqualStrings("world", result[1]);
}

test "Whitespace: splits on carriage return" {
    const allocator = std.testing.allocator;
    var ws = Whitespace{};
    const pt = ws.preTokenizer();

    const result = try pt.preTokenize(allocator, "hello\rworld");
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 2), result.len);
    try std.testing.expectEqualStrings("hello", result[0]);
    try std.testing.expectEqualStrings("world", result[1]);
}

test "Whitespace: multiple consecutive spaces" {
    const allocator = std.testing.allocator;
    var ws = Whitespace{};
    const pt = ws.preTokenizer();

    const result = try pt.preTokenize(allocator, "hello   world");
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 2), result.len);
    try std.testing.expectEqualStrings("hello", result[0]);
    try std.testing.expectEqualStrings("world", result[1]);
}

test "Whitespace: empty string returns empty" {
    const allocator = std.testing.allocator;
    var ws = Whitespace{};
    const pt = ws.preTokenizer();

    const result = try pt.preTokenize(allocator, "");
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 0), result.len);
}

test "Whitespace: single word" {
    const allocator = std.testing.allocator;
    var ws = Whitespace{};
    const pt = ws.preTokenizer();

    const result = try pt.preTokenize(allocator, "hello");
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 1), result.len);
    try std.testing.expectEqualStrings("hello", result[0]);
}

test "Whitespace: leading and trailing whitespace" {
    const allocator = std.testing.allocator;
    var ws = Whitespace{};
    const pt = ws.preTokenizer();

    const result = try pt.preTokenize(allocator, "  hello world  ");
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 2), result.len);
    try std.testing.expectEqualStrings("hello", result[0]);
    try std.testing.expectEqualStrings("world", result[1]);
}

test "Whitespace: mixed whitespace types" {
    const allocator = std.testing.allocator;
    var ws = Whitespace{};
    const pt = ws.preTokenizer();

    const result = try pt.preTokenize(allocator, "a \t\n\r b");
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 2), result.len);
    try std.testing.expectEqualStrings("a", result[0]);
    try std.testing.expectEqualStrings("b", result[1]);
}

test "Whitespace: preserves unicode content" {
    const allocator = std.testing.allocator;
    var ws = Whitespace{};
    const pt = ws.preTokenizer();

    const result = try pt.preTokenize(allocator, "hello \xe4\xb8\x96\xe7\x95\x8c");
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 2), result.len);
    try std.testing.expectEqualStrings("hello", result[0]);
    try std.testing.expectEqualStrings("\xe4\xb8\x96\xe7\x95\x8c", result[1]);
}

test "BertPreTokenizer: splits on space" {
    const allocator = std.testing.allocator;
    var bert = BertPreTokenizer{};
    const pt = bert.preTokenizer();

    const result = try pt.preTokenize(allocator, "hello world");
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 2), result.len);
    try std.testing.expectEqualStrings("hello", result[0]);
    try std.testing.expectEqualStrings("world", result[1]);
}

test "BertPreTokenizer: period as separate token" {
    const allocator = std.testing.allocator;
    var bert = BertPreTokenizer{};
    const pt = bert.preTokenizer();

    const result = try pt.preTokenize(allocator, "hello.");
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 2), result.len);
    try std.testing.expectEqualStrings("hello", result[0]);
    try std.testing.expectEqualStrings(".", result[1]);
}

test "BertPreTokenizer: comma as separate token" {
    const allocator = std.testing.allocator;
    var bert = BertPreTokenizer{};
    const pt = bert.preTokenizer();

    const result = try pt.preTokenize(allocator, "a,b");
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 3), result.len);
    try std.testing.expectEqualStrings("a", result[0]);
    try std.testing.expectEqualStrings(",", result[1]);
    try std.testing.expectEqualStrings("b", result[2]);
}

test "BertPreTokenizer: exclamation as separate token" {
    const allocator = std.testing.allocator;
    var bert = BertPreTokenizer{};
    const pt = bert.preTokenizer();

    const result = try pt.preTokenize(allocator, "wow!");
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 2), result.len);
    try std.testing.expectEqualStrings("wow", result[0]);
    try std.testing.expectEqualStrings("!", result[1]);
}

test "BertPreTokenizer: question mark as separate token" {
    const allocator = std.testing.allocator;
    var bert = BertPreTokenizer{};
    const pt = bert.preTokenizer();

    const result = try pt.preTokenize(allocator, "what?");
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 2), result.len);
    try std.testing.expectEqualStrings("what", result[0]);
    try std.testing.expectEqualStrings("?", result[1]);
}

test "BertPreTokenizer: brackets as separate tokens" {
    const allocator = std.testing.allocator;
    var bert = BertPreTokenizer{};
    const pt = bert.preTokenizer();

    const result = try pt.preTokenize(allocator, "[hello]");
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 3), result.len);
    try std.testing.expectEqualStrings("[", result[0]);
    try std.testing.expectEqualStrings("hello", result[1]);
    try std.testing.expectEqualStrings("]", result[2]);
}

test "BertPreTokenizer: mixed punctuation and words" {
    const allocator = std.testing.allocator;
    var bert = BertPreTokenizer{};
    const pt = bert.preTokenizer();

    // "Hello, world!" -> ["Hello", ",", "world", "!"] (4 tokens, space consumed)
    const result = try pt.preTokenize(allocator, "Hello, world!");
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 4), result.len);
    try std.testing.expectEqualStrings("Hello", result[0]);
    try std.testing.expectEqualStrings(",", result[1]);
    try std.testing.expectEqualStrings("world", result[2]);
    try std.testing.expectEqualStrings("!", result[3]);
}

test "BertPreTokenizer: handles empty string" {
    const allocator = std.testing.allocator;
    var bert = BertPreTokenizer{};
    const pt = bert.preTokenizer();

    const result = try pt.preTokenize(allocator, "");
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 0), result.len);
}

test "BertPreTokenizer: consecutive punctuation" {
    const allocator = std.testing.allocator;
    var bert = BertPreTokenizer{};
    const pt = bert.preTokenizer();

    const result = try pt.preTokenize(allocator, "...");
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 3), result.len);
    try std.testing.expectEqualStrings(".", result[0]);
    try std.testing.expectEqualStrings(".", result[1]);
    try std.testing.expectEqualStrings(".", result[2]);
}

test "BertPreTokenizer: punctuation at start" {
    const allocator = std.testing.allocator;
    var bert = BertPreTokenizer{};
    const pt = bert.preTokenizer();

    const result = try pt.preTokenize(allocator, "!hello");
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 2), result.len);
    try std.testing.expectEqualStrings("!", result[0]);
    try std.testing.expectEqualStrings("hello", result[1]);
}

test "BertPreTokenizer: punctuation at end" {
    const allocator = std.testing.allocator;
    var bert = BertPreTokenizer{};
    const pt = bert.preTokenizer();

    const result = try pt.preTokenize(allocator, "hello!");
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 2), result.len);
    try std.testing.expectEqualStrings("hello", result[0]);
    try std.testing.expectEqualStrings("!", result[1]);
}

test "BertPreTokenizer: isPunctuation covers all ranges" {
    // ASCII punctuation ranges: 33-47, 58-64, 91-96, 123-126
    const allocator = std.testing.allocator;
    var bert = BertPreTokenizer{};
    const pt = bert.preTokenizer();

    // Test boundary characters from each range
    const result = try pt.preTokenize(allocator, "!/:@[`{~");
    defer allocator.free(result);

    // Each character should be its own token
    try std.testing.expectEqual(@as(usize, 8), result.len);
    try std.testing.expectEqualStrings("!", result[0]); // 33
    try std.testing.expectEqualStrings("/", result[1]); // 47
    try std.testing.expectEqualStrings(":", result[2]); // 58
    try std.testing.expectEqualStrings("@", result[3]); // 64
    try std.testing.expectEqualStrings("[", result[4]); // 91
    try std.testing.expectEqualStrings("`", result[5]); // 96
    try std.testing.expectEqualStrings("{", result[6]); // 123
    try std.testing.expectEqualStrings("~", result[7]); // 126
}

test "ByteLevel: splits on whitespace" {
    const allocator = std.testing.allocator;
    var bl = ByteLevel{};
    const pt = bl.preTokenizer();

    const result = try pt.preTokenize(allocator, "hello world");
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 2), result.len);
    try std.testing.expectEqualStrings("hello", result[0]);
    try std.testing.expectEqualStrings("world", result[1]);
}

test "ByteLevel: empty string" {
    const allocator = std.testing.allocator;
    var bl = ByteLevel{};
    const pt = bl.preTokenizer();

    const result = try pt.preTokenize(allocator, "");
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 0), result.len);
}

test "ByteLevel: single word" {
    const allocator = std.testing.allocator;
    var bl = ByteLevel{};
    const pt = bl.preTokenizer();

    const result = try pt.preTokenize(allocator, "hello");
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 1), result.len);
    try std.testing.expectEqualStrings("hello", result[0]);
}

test "ByteLevel: preserves UTF-8" {
    const allocator = std.testing.allocator;
    var bl = ByteLevel{};
    const pt = bl.preTokenizer();

    const result = try pt.preTokenize(allocator, "\xe4\xb8\x96\xe7\x95\x8c");
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 1), result.len);
    try std.testing.expectEqualStrings("\xe4\xb8\x96\xe7\x95\x8c", result[0]);
}

test "ByteLevel: multiple words" {
    const allocator = std.testing.allocator;
    var bl = ByteLevel{};
    const pt = bl.preTokenizer();

    const result = try pt.preTokenize(allocator, "a b c d");
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 4), result.len);
}

test "Sequence: empty list returns input wrapped" {
    const allocator = std.testing.allocator;
    var seq = try Sequence.init(allocator, &.{});
    defer seq.deinit();
    const pt = seq.preTokenizer();

    const result = try pt.preTokenize(allocator, "hello world");
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 1), result.len);
    try std.testing.expectEqualStrings("hello world", result[0]);
}

test "Sequence: single pre-tokenizer" {
    const allocator = std.testing.allocator;
    var ws = Whitespace{};

    var seq = try Sequence.init(allocator, &.{ws.preTokenizer()});
    defer seq.deinit();
    const pt = seq.preTokenizer();

    const result = try pt.preTokenize(allocator, "hello world");
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 2), result.len);
    try std.testing.expectEqualStrings("hello", result[0]);
    try std.testing.expectEqualStrings("world", result[1]);
}

test "Sequence: chained Whitespace then BertPreTokenizer" {
    const allocator = std.testing.allocator;
    var ws = Whitespace{};
    var bert = BertPreTokenizer{};

    var seq = try Sequence.init(allocator, &.{ ws.preTokenizer(), bert.preTokenizer() });
    defer seq.deinit();
    const pt = seq.preTokenizer();

    // First Whitespace splits on spaces, then BERT splits each token on punctuation
    const result = try pt.preTokenize(allocator, "hello! world.");
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 4), result.len);
    try std.testing.expectEqualStrings("hello", result[0]);
    try std.testing.expectEqualStrings("!", result[1]);
    try std.testing.expectEqualStrings("world", result[2]);
    try std.testing.expectEqualStrings(".", result[3]);
}

test "Sequence: deinit via interface" {
    const allocator = std.testing.allocator;
    var ws = Whitespace{};

    var seq = try Sequence.init(allocator, &.{ws.preTokenizer()});
    var pt = seq.preTokenizer();

    const result = try pt.preTokenize(allocator, "a b");
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 2), result.len);

    // deinit via interface
    pt.deinit();
}

test "PreTokenizer: deinit with null function is safe" {
    var ws = Whitespace{};
    var pt = ws.preTokenizer();

    // deinit_fn is null for Whitespace
    pt.deinit();
}

test "Sequence: multiple passes accumulate tokens" {
    const allocator = std.testing.allocator;
    var ws1 = Whitespace{};
    var ws2 = Whitespace{};

    // Two whitespace passes should give same result (idempotent)
    var seq = try Sequence.init(allocator, &.{ ws1.preTokenizer(), ws2.preTokenizer() });
    defer seq.deinit();
    const pt = seq.preTokenizer();

    const result = try pt.preTokenize(allocator, "a b c");
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 3), result.len);
}
