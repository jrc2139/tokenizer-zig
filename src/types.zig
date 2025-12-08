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
