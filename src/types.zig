//! Common types used across the tokenizer library

/// Byte offset range in the original text
pub const Offset = struct {
    start: u32,
    end: u32,

    pub fn init(start: u32, end: u32) Offset {
        return .{ .start = start, .end = end };
    }
};

/// Added token configuration
pub const AddedToken = struct {
    content: []const u8,
    id: ?u32 = null,
    single_word: bool = false,
    lstrip: bool = false,
    rstrip: bool = false,
    normalized: bool = true,
    special: bool = false,

    pub fn init(content: []const u8, special: bool) AddedToken {
        return .{ .content = content, .special = special };
    }

    pub fn withId(content: []const u8, id: u32, special: bool) AddedToken {
        return .{ .content = content, .id = id, .special = special };
    }
};

/// Padding direction
pub const PaddingDirection = enum {
    left,
    right,
};

/// Padding parameters
pub const PaddingParams = struct {
    length: ?usize = null,
    pad_id: u32 = 0,
    pad_type_id: u32 = 0,
    pad_token: []const u8 = "[PAD]",
    direction: PaddingDirection = .right,
};

/// Truncation strategy
pub const TruncationStrategy = enum {
    longest_first,
    only_first,
    only_second,
};

/// Truncation parameters
pub const TruncationParams = struct {
    max_length: usize = 512,
    strategy: TruncationStrategy = .longest_first,
    stride: usize = 0,
};

// ============================================================================
// Unit Tests
// ============================================================================

test "Offset init and fields" {
    const offset = Offset.init(10, 25);
    try @import("std").testing.expectEqual(@as(u32, 10), offset.start);
    try @import("std").testing.expectEqual(@as(u32, 25), offset.end);
}

test "Offset zero range" {
    const offset = Offset.init(0, 0);
    try @import("std").testing.expectEqual(@as(u32, 0), offset.start);
    try @import("std").testing.expectEqual(@as(u32, 0), offset.end);
}

test "AddedToken init basic" {
    const token = AddedToken.init("hello", false);
    try @import("std").testing.expectEqualStrings("hello", token.content);
    try @import("std").testing.expect(!token.special);
    try @import("std").testing.expect(token.id == null);
}

test "AddedToken init special" {
    const token = AddedToken.init("[CLS]", true);
    try @import("std").testing.expectEqualStrings("[CLS]", token.content);
    try @import("std").testing.expect(token.special);
}

test "AddedToken withId" {
    const token = AddedToken.withId("[PAD]", 0, true);
    try @import("std").testing.expectEqualStrings("[PAD]", token.content);
    try @import("std").testing.expectEqual(@as(u32, 0), token.id.?);
    try @import("std").testing.expect(token.special);
}

test "AddedToken default fields" {
    const token = AddedToken.init("test", false);
    try @import("std").testing.expect(!token.single_word);
    try @import("std").testing.expect(!token.lstrip);
    try @import("std").testing.expect(!token.rstrip);
    try @import("std").testing.expect(token.normalized);
}

test "PaddingParams defaults" {
    const params = PaddingParams{};
    try @import("std").testing.expect(params.length == null);
    try @import("std").testing.expectEqual(@as(u32, 0), params.pad_id);
    try @import("std").testing.expectEqual(@as(u32, 0), params.pad_type_id);
    try @import("std").testing.expectEqualStrings("[PAD]", params.pad_token);
    try @import("std").testing.expectEqual(PaddingDirection.right, params.direction);
}

test "PaddingParams custom" {
    const params = PaddingParams{
        .length = 128,
        .pad_id = 1,
        .pad_type_id = 2,
        .pad_token = "<pad>",
        .direction = .left,
    };
    try @import("std").testing.expectEqual(@as(usize, 128), params.length.?);
    try @import("std").testing.expectEqual(@as(u32, 1), params.pad_id);
    try @import("std").testing.expectEqual(PaddingDirection.left, params.direction);
}

test "TruncationParams defaults" {
    const params = TruncationParams{};
    try @import("std").testing.expectEqual(@as(usize, 512), params.max_length);
    try @import("std").testing.expectEqual(TruncationStrategy.longest_first, params.strategy);
    try @import("std").testing.expectEqual(@as(usize, 0), params.stride);
}

test "TruncationParams custom" {
    const params = TruncationParams{
        .max_length = 256,
        .strategy = .only_first,
        .stride = 16,
    };
    try @import("std").testing.expectEqual(@as(usize, 256), params.max_length);
    try @import("std").testing.expectEqual(TruncationStrategy.only_first, params.strategy);
    try @import("std").testing.expectEqual(@as(usize, 16), params.stride);
}
