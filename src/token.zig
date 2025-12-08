//! Token type representing a single tokenized unit

const std = @import("std");
const lib = @import("lib.zig");

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
