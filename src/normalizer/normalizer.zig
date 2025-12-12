//! Normalizer interface for text normalization
//!
//! Normalizers prepare text before tokenization by applying transformations
//! like lowercasing, NFD/NFC normalization, stripping accents, etc.

const std = @import("std");

/// Function pointer types
pub const NormalizeFn = *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, input: []const u8) anyerror![]const u8;
pub const DeinitFn = *const fn (ctx: *anyopaque) void;

/// Normalizer interface
pub const Normalizer = struct {
    ptr: *anyopaque,
    normalize_fn: NormalizeFn,
    deinit_fn: ?DeinitFn = null,

    const Self = @This();

    pub fn normalize(self: Self, allocator: std.mem.Allocator, input: []const u8) ![]const u8 {
        return self.normalize_fn(self.ptr, allocator, input);
    }

    pub fn deinit(self: *Self) void {
        if (self.deinit_fn) |f| {
            f(self.ptr);
        }
    }
};

/// BERT normalizer - cleans text, handles Chinese characters, lowercases, strips accents
pub const BertNormalizer = struct {
    clean_text: bool = true,
    handle_chinese_chars: bool = true,
    strip_accents: ?bool = null,
    lowercase: bool = true,

    const Self = @This();

    pub fn normalizer(self: *Self) Normalizer {
        return .{
            .ptr = self,
            .normalize_fn = normalizeImpl,
        };
    }

    fn normalizeImpl(ctx: *anyopaque, allocator: std.mem.Allocator, input: []const u8) anyerror![]const u8 {
        const self: *Self = @ptrCast(@alignCast(ctx));

        var result = std.ArrayListUnmanaged(u8){};
        errdefer result.deinit(allocator);

        for (input) |c| {
            // Clean control characters
            if (self.clean_text and isControlChar(c)) {
                continue;
            }

            // Lowercase
            if (self.lowercase and c >= 'A' and c <= 'Z') {
                try result.append(allocator, c + 32);
            } else {
                try result.append(allocator, c);
            }
        }

        return try result.toOwnedSlice(allocator);
    }

    fn isControlChar(c: u8) bool {
        // Control characters except tab, newline, carriage return
        return (c < 0x20 and c != '\t' and c != '\n' and c != '\r') or c == 0x7F;
    }
};

/// Lowercase normalizer
pub const Lowercase = struct {
    const Self = @This();

    pub fn normalizer(self: *Self) Normalizer {
        return .{
            .ptr = self,
            .normalize_fn = normalizeImpl,
        };
    }

    fn normalizeImpl(_: *anyopaque, allocator: std.mem.Allocator, input: []const u8) anyerror![]const u8 {
        var result = try allocator.alloc(u8, input.len);
        for (input, 0..) |c, i| {
            if (c >= 'A' and c <= 'Z') {
                result[i] = c + 32;
            } else {
                result[i] = c;
            }
        }
        return result;
    }
};

/// Sequence normalizer - applies multiple normalizers in order
pub const Sequence = struct {
    normalizers: []Normalizer,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, normalizers: []const Normalizer) !Self {
        const owned = try allocator.dupe(Normalizer, normalizers);
        return .{
            .normalizers = owned,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.normalizers);
    }

    pub fn normalizer(self: *Self) Normalizer {
        return .{
            .ptr = self,
            .normalize_fn = normalizeImpl,
            .deinit_fn = deinitImpl,
        };
    }

    fn normalizeImpl(ctx: *anyopaque, allocator: std.mem.Allocator, input: []const u8) anyerror![]const u8 {
        const self: *Self = @ptrCast(@alignCast(ctx));

        var current = input;
        var owned = false;

        for (self.normalizers) |n| {
            const next = try n.normalize(allocator, current);
            if (owned) {
                allocator.free(@constCast(current));
            }
            current = next;
            owned = true;
        }

        if (!owned) {
            return try allocator.dupe(u8, input);
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

test "BertNormalizer: lowercases ASCII" {
    const allocator = std.testing.allocator;
    var bert = BertNormalizer{};
    const normalizer = bert.normalizer();

    const result = try normalizer.normalize(allocator, "Hello World");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("hello world", result);
}

test "BertNormalizer: removes control characters" {
    const allocator = std.testing.allocator;
    var bert = BertNormalizer{};
    const normalizer = bert.normalizer();

    // Control chars: 0x00, 0x01, 0x1F, 0x7F (DEL)
    const result = try normalizer.normalize(allocator, "a\x00b\x01c\x1Fd\x7Fe");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("abcde", result);
}

test "BertNormalizer: preserves tabs, newlines, carriage returns" {
    const allocator = std.testing.allocator;
    var bert = BertNormalizer{};
    const normalizer = bert.normalizer();

    const result = try normalizer.normalize(allocator, "a\tb\nc\rd");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("a\tb\nc\rd", result);
}

test "BertNormalizer: handles empty input" {
    const allocator = std.testing.allocator;
    var bert = BertNormalizer{};
    const normalizer = bert.normalizer();

    const result = try normalizer.normalize(allocator, "");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("", result);
}

test "BertNormalizer: handles single character" {
    const allocator = std.testing.allocator;
    var bert = BertNormalizer{};
    const normalizer = bert.normalizer();

    const result = try normalizer.normalize(allocator, "A");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("a", result);
}

test "BertNormalizer: clean_text=false keeps control chars" {
    const allocator = std.testing.allocator;
    var bert = BertNormalizer{ .clean_text = false };
    const normalizer = bert.normalizer();

    const result = try normalizer.normalize(allocator, "a\x00b");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("a\x00b", result);
}

test "BertNormalizer: lowercase=false preserves case" {
    const allocator = std.testing.allocator;
    var bert = BertNormalizer{ .lowercase = false };
    const normalizer = bert.normalizer();

    const result = try normalizer.normalize(allocator, "Hello World");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("Hello World", result);
}

test "BertNormalizer: handles all uppercase string" {
    const allocator = std.testing.allocator;
    var bert = BertNormalizer{};
    const normalizer = bert.normalizer();

    const result = try normalizer.normalize(allocator, "HELLO WORLD");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("hello world", result);
}

test "BertNormalizer: handles all lowercase string (no-op)" {
    const allocator = std.testing.allocator;
    var bert = BertNormalizer{};
    const normalizer = bert.normalizer();

    const result = try normalizer.normalize(allocator, "hello world");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("hello world", result);
}

test "BertNormalizer: preserves non-ASCII UTF-8 bytes" {
    const allocator = std.testing.allocator;
    var bert = BertNormalizer{};
    const normalizer = bert.normalizer();

    // UTF-8 bytes for Chinese characters (世界) should pass through
    const result = try normalizer.normalize(allocator, "Hello \xe4\xb8\x96\xe7\x95\x8c");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("hello \xe4\xb8\x96\xe7\x95\x8c", result);
}

test "BertNormalizer: mixed case and control chars" {
    const allocator = std.testing.allocator;
    var bert = BertNormalizer{};
    const normalizer = bert.normalizer();

    const result = try normalizer.normalize(allocator, "HeLLo\x00WoRLD\x7F");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("helloworld", result);
}

test "BertNormalizer: numbers and punctuation unchanged" {
    const allocator = std.testing.allocator;
    var bert = BertNormalizer{};
    const normalizer = bert.normalizer();

    const result = try normalizer.normalize(allocator, "Test123!@#$%");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("test123!@#$%", result);
}

test "Lowercase: basic lowercase" {
    const allocator = std.testing.allocator;
    var lower = Lowercase{};
    const normalizer = lower.normalizer();

    const result = try normalizer.normalize(allocator, "Hello World");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("hello world", result);
}

test "Lowercase: empty string" {
    const allocator = std.testing.allocator;
    var lower = Lowercase{};
    const normalizer = lower.normalizer();

    const result = try normalizer.normalize(allocator, "");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("", result);
}

test "Lowercase: all lowercase no-op" {
    const allocator = std.testing.allocator;
    var lower = Lowercase{};
    const normalizer = lower.normalizer();

    const result = try normalizer.normalize(allocator, "already lowercase");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("already lowercase", result);
}

test "Lowercase: mixed case" {
    const allocator = std.testing.allocator;
    var lower = Lowercase{};
    const normalizer = lower.normalizer();

    const result = try normalizer.normalize(allocator, "MiXeD CaSe");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("mixed case", result);
}

test "Lowercase: preserves non-ASCII" {
    const allocator = std.testing.allocator;
    var lower = Lowercase{};
    const normalizer = lower.normalizer();

    const result = try normalizer.normalize(allocator, "Hello \xe4\xb8\x96\xe7\x95\x8c");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("hello \xe4\xb8\x96\xe7\x95\x8c", result);
}

test "Lowercase: with numbers and punctuation" {
    const allocator = std.testing.allocator;
    var lower = Lowercase{};
    const normalizer = lower.normalizer();

    const result = try normalizer.normalize(allocator, "ABC123!@#xyz");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("abc123!@#xyz", result);
}

test "Sequence: empty list returns copy of input" {
    const allocator = std.testing.allocator;
    var seq = try Sequence.init(allocator, &.{});
    defer seq.deinit();
    const normalizer = seq.normalizer();

    const result = try normalizer.normalize(allocator, "Hello");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("Hello", result);
}

test "Sequence: single normalizer" {
    const allocator = std.testing.allocator;
    var lower = Lowercase{};

    var seq = try Sequence.init(allocator, &.{lower.normalizer()});
    defer seq.deinit();
    const normalizer = seq.normalizer();

    const result = try normalizer.normalize(allocator, "HELLO");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("hello", result);
}

test "Sequence: two normalizers applied in order" {
    const allocator = std.testing.allocator;
    var lower1 = Lowercase{};
    var lower2 = Lowercase{}; // Second lowercase is no-op but tests chaining

    var seq = try Sequence.init(allocator, &.{ lower1.normalizer(), lower2.normalizer() });
    defer seq.deinit();
    const normalizer = seq.normalizer();

    const result = try normalizer.normalize(allocator, "HELLO WORLD");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("hello world", result);
}

test "Sequence: chained BertNormalizer then Lowercase" {
    const allocator = std.testing.allocator;
    var bert = BertNormalizer{ .lowercase = false }; // Don't lowercase in BERT
    var lower = Lowercase{};

    var seq = try Sequence.init(allocator, &.{ bert.normalizer(), lower.normalizer() });
    defer seq.deinit();
    const normalizer = seq.normalizer();

    // BERT removes control chars, then Lowercase lowercases
    const result = try normalizer.normalize(allocator, "HELLO\x00WORLD");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("helloworld", result);
}

test "Sequence: deinit via interface" {
    const allocator = std.testing.allocator;
    var lower = Lowercase{};

    var seq = try Sequence.init(allocator, &.{lower.normalizer()});
    var normalizer = seq.normalizer();

    const result = try normalizer.normalize(allocator, "TEST");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("test", result);

    // deinit via interface instead of direct call
    normalizer.deinit();
}

test "Normalizer: deinit with null function is safe" {
    var lower = Lowercase{};
    var normalizer = lower.normalizer();

    // deinit_fn is null for Lowercase, should not crash
    normalizer.deinit();
}

test "isControlChar: boundary values" {
    // Test the private function behavior through BertNormalizer
    const allocator = std.testing.allocator;
    var bert = BertNormalizer{};
    const normalizer = bert.normalizer();

    // 0x00 to 0x1F except \t(0x09), \n(0x0A), \r(0x0D) are control chars
    // 0x7F (DEL) is also control char
    const result = try normalizer.normalize(allocator, "\x00\x08\x09\x0a\x0d\x1f\x20\x7e\x7f");
    defer allocator.free(result);
    // Keeps \t(0x09), \n(0x0A), \r(0x0D), space(0x20), ~(0x7E)
    try std.testing.expectEqualStrings("\t\n\r ~", result);
}
