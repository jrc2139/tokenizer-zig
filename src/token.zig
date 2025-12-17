//! Token type representing a single tokenized unit

const std = @import("std");
const lib = @import("lib.zig");

// ============================================================================
// SpanToken - Zero-Copy Token Representation
// ============================================================================

/// Flags for SpanToken packed into a single byte
pub const SpanTokenFlags = packed struct(u8) {
    is_special: bool = false,
    is_continuation: bool = false, // WordPiece ##prefix
    is_padding: bool = false,
    _reserved: u5 = 0,
};

/// Zero-copy token that stores offsets into original input rather than string copies.
/// Size: 16 bytes (id:4 + start:4 + end:4 + type_id:1 + flags:1 + pad:2)
pub const SpanToken = struct {
    /// Token ID from vocabulary
    id: u32,
    /// Byte offset into original input (start of token)
    start: u32,
    /// Byte offset into original input (end of token, exclusive)
    end: u32,
    /// Type ID for sentence pair encoding (0 = first sentence, 1 = second)
    type_id: u8 = 0,
    /// Token flags
    flags: SpanTokenFlags = .{},
    /// Padding for alignment
    _pad: u16 = 0,

    const Self = @This();

    /// Create a new SpanToken
    pub fn init(id: u32, start: u32, end: u32) Self {
        return .{ .id = id, .start = start, .end = end };
    }

    /// Create a SpanToken with flags
    pub fn initWithFlags(id: u32, start: u32, end: u32, flags: SpanTokenFlags) Self {
        return .{ .id = id, .start = start, .end = end, .flags = flags };
    }

    /// Create a special token (e.g., [CLS], [SEP])
    pub fn initSpecial(id: u32, start: u32, end: u32) Self {
        return .{ .id = id, .start = start, .end = end, .flags = .{ .is_special = true } };
    }

    /// Create a padding token
    pub fn initPadding(pad_id: u32) Self {
        return .{ .id = pad_id, .start = 0, .end = 0, .flags = .{ .is_padding = true } };
    }

    /// Get the byte length of this token
    pub inline fn len(self: Self) u32 {
        return self.end - self.start;
    }

    /// Get the token string from the original input (zero-copy)
    pub inline fn slice(self: Self, input: []const u8) []const u8 {
        return input[self.start..self.end];
    }

    /// Convert to legacy Token type (requires vocab for string lookup)
    pub fn toToken(self: Self, input: []const u8) Token {
        return Token.init(self.id, self.slice(input), self.start, self.end);
    }
};

// Compile-time size verification
comptime {
    // Ensure SpanToken is exactly 16 bytes for cache-friendly access
    std.debug.assert(@sizeOf(SpanToken) == 16);
}

// ============================================================================
// Token - Original Implementation (Backward Compatibility)
// ============================================================================

/// A single token with its ID, string value, and byte offsets
pub const Token = struct {
    id: u32,
    value: []const u8,
    offset: lib.Offset,

    const Self = @This();

    pub fn init(id: u32, value: []const u8, start: u32, end: u32) Self {
        return .{
            .id = id,
            .value = value,
            .offset = lib.Offset.init(start, end),
        };
    }
};

// ============================================================================
// Unit Tests
// ============================================================================

test "Token init basic" {
    const token = Token.init(42, "hello", 0, 5);
    try std.testing.expectEqual(@as(u32, 42), token.id);
    try std.testing.expectEqualStrings("hello", token.value);
    try std.testing.expectEqual(@as(u32, 0), token.offset.start);
    try std.testing.expectEqual(@as(u32, 5), token.offset.end);
}

test "Token with unicode" {
    const token = Token.init(100, "世界", 6, 12);
    try std.testing.expectEqual(@as(u32, 100), token.id);
    try std.testing.expectEqualStrings("世界", token.value);
    try std.testing.expectEqual(@as(u32, 6), token.offset.start);
    try std.testing.expectEqual(@as(u32, 12), token.offset.end);
}

test "Token empty value" {
    const token = Token.init(0, "", 0, 0);
    try std.testing.expectEqual(@as(u32, 0), token.id);
    try std.testing.expectEqual(@as(usize, 0), token.value.len);
}

test "Token special token" {
    const token = Token.init(101, "[CLS]", 0, 0);
    try std.testing.expectEqual(@as(u32, 101), token.id);
    try std.testing.expectEqualStrings("[CLS]", token.value);
}

test "Token wordpiece subword" {
    const token = Token.init(200, "##ing", 4, 7);
    try std.testing.expectEqual(@as(u32, 200), token.id);
    try std.testing.expectEqualStrings("##ing", token.value);
}

// ============================================================================
// SpanToken Unit Tests
// ============================================================================

test "SpanToken size is 16 bytes" {
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(SpanToken));
}

test "SpanToken init basic" {
    const token = SpanToken.init(42, 0, 5);
    try std.testing.expectEqual(@as(u32, 42), token.id);
    try std.testing.expectEqual(@as(u32, 0), token.start);
    try std.testing.expectEqual(@as(u32, 5), token.end);
    try std.testing.expectEqual(@as(u8, 0), token.type_id);
    try std.testing.expect(!token.flags.is_special);
    try std.testing.expect(!token.flags.is_padding);
}

test "SpanToken slice" {
    const input = "hello world";
    const token = SpanToken.init(1, 0, 5);
    try std.testing.expectEqualStrings("hello", token.slice(input));

    const token2 = SpanToken.init(2, 6, 11);
    try std.testing.expectEqualStrings("world", token2.slice(input));
}

test "SpanToken len" {
    const token = SpanToken.init(1, 10, 25);
    try std.testing.expectEqual(@as(u32, 15), token.len());
}

test "SpanToken special flag" {
    const token = SpanToken.initSpecial(101, 0, 5);
    try std.testing.expect(token.flags.is_special);
    try std.testing.expect(!token.flags.is_padding);
}

test "SpanToken padding flag" {
    const token = SpanToken.initPadding(0);
    try std.testing.expect(token.flags.is_padding);
    try std.testing.expect(!token.flags.is_special);
    try std.testing.expectEqual(@as(u32, 0), token.start);
    try std.testing.expectEqual(@as(u32, 0), token.end);
}

test "SpanToken toToken conversion" {
    const input = "hello world";
    const span = SpanToken.init(42, 0, 5);
    const token = span.toToken(input);

    try std.testing.expectEqual(@as(u32, 42), token.id);
    try std.testing.expectEqualStrings("hello", token.value);
    try std.testing.expectEqual(@as(u32, 0), token.offset.start);
    try std.testing.expectEqual(@as(u32, 5), token.offset.end);
}

test "SpanToken with flags" {
    const token = SpanToken.initWithFlags(1, 0, 5, .{ .is_continuation = true });
    try std.testing.expect(token.flags.is_continuation);
    try std.testing.expect(!token.flags.is_special);
}
