//! Token type representing a single tokenized unit

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
