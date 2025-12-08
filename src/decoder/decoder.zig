//! Decoder interface for decoding tokens back to text
//!
//! Decoders handle post-processing after token strings are joined,
//! like removing byte-level encoding or subword prefixes.

const std = @import("std");

/// Function pointer types
pub const DecodeFn = *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, tokens: []const u8) anyerror![]u8;
pub const DeinitFn = *const fn (ctx: *anyopaque) void;

/// Decoder interface
pub const Decoder = struct {
    ptr: *anyopaque,
    decode_fn: DecodeFn,
    deinit_fn: ?DeinitFn = null,

    const Self = @This();

    pub fn decode(self: Self, allocator: std.mem.Allocator, tokens: []const u8) ![]u8 {
        return self.decode_fn(self.ptr, allocator, tokens);
    }

    pub fn deinit(self: *Self) void {
        if (self.deinit_fn) |f| {
            f(self.ptr);
        }
    }
};

/// WordPiece decoder - removes ## prefix from subword tokens
pub const WordPieceDecoder = struct {
    prefix: []const u8 = "##",
    cleanup: bool = true,

    const Self = @This();

    pub fn decoder(self: *Self) Decoder {
        return .{
            .ptr = self,
            .decode_fn = decodeImpl,
        };
    }

    fn decodeImpl(ctx: *anyopaque, allocator: std.mem.Allocator, tokens: []const u8) anyerror![]u8 {
        const self: *Self = @ptrCast(@alignCast(ctx));

        var result = std.ArrayListUnmanaged(u8){};
        errdefer result.deinit(allocator);

        var i: usize = 0;
        while (i < tokens.len) {
            // Check for prefix
            if (std.mem.startsWith(u8, tokens[i..], self.prefix)) {
                i += self.prefix.len;
            } else if (result.items.len > 0) {
                // Add space between words (not before first)
                try result.append(allocator, ' ');
            }

            // Copy until next prefix or end
            const start = i;
            while (i < tokens.len) {
                if (std.mem.startsWith(u8, tokens[i..], self.prefix)) {
                    break;
                }
                i += 1;
            }
            try result.appendSlice(allocator, tokens[start..i]);
        }

        return try result.toOwnedSlice(allocator);
    }
};

/// BPE decoder - handles byte-level decoding
pub const BPEDecoder = struct {
    suffix: []const u8 = "</w>",

    const Self = @This();

    pub fn decoder(self: *Self) Decoder {
        return .{
            .ptr = self,
            .decode_fn = decodeImpl,
        };
    }

    fn decodeImpl(ctx: *anyopaque, allocator: std.mem.Allocator, tokens: []const u8) anyerror![]u8 {
        const self: *Self = @ptrCast(@alignCast(ctx));
        _ = self;

        // Simple pass-through for now
        // Full implementation would handle byte-to-unicode mapping
        return try allocator.dupe(u8, tokens);
    }
};

/// ByteLevel decoder - converts byte tokens back to characters
pub const ByteLevelDecoder = struct {
    const Self = @This();

    pub fn decoder(self: *Self) Decoder {
        return .{
            .ptr = self,
            .decode_fn = decodeImpl,
        };
    }

    fn decodeImpl(_: *anyopaque, allocator: std.mem.Allocator, tokens: []const u8) anyerror![]u8 {
        // ByteLevel uses a unicode-to-byte mapping that needs to be inverted
        // For now, simple pass-through
        return try allocator.dupe(u8, tokens);
    }
};

/// Sequence decoder - applies multiple decoders
pub const Sequence = struct {
    decoders: []Decoder,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, decoders: []const Decoder) !Self {
        const owned = try allocator.dupe(Decoder, decoders);
        return .{
            .decoders = owned,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.decoders);
    }

    pub fn decoder(self: *Self) Decoder {
        return .{
            .ptr = self,
            .decode_fn = decodeImpl,
            .deinit_fn = deinitImpl,
        };
    }

    fn decodeImpl(ctx: *anyopaque, allocator: std.mem.Allocator, input: []const u8) anyerror![]u8 {
        const self: *Self = @ptrCast(@alignCast(ctx));

        var current = try allocator.dupe(u8, input);

        for (self.decoders) |d| {
            const next = try d.decode(allocator, current);
            allocator.free(current);
            current = next;
        }

        return current;
    }

    fn deinitImpl(ctx: *anyopaque) void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        self.deinit();
    }
};
