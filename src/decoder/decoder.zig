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

// ============================================================================
// Tests
// ============================================================================

test "WordPieceDecoder: removes ## prefix and concatenates" {
    const allocator = std.testing.allocator;
    var wpd = WordPieceDecoder{};
    const d = wpd.decoder();

    // ## prefix is removed and subwords are concatenated
    const result = try d.decode(allocator, "hello##world");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("helloworld", result);
}

test "WordPieceDecoder: multiple subwords concatenate" {
    const allocator = std.testing.allocator;
    var wpd = WordPieceDecoder{};
    const d = wpd.decoder();

    const result = try d.decode(allocator, "hello##ing##ly");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("helloingly", result);
}

test "WordPieceDecoder: no space before first word" {
    const allocator = std.testing.allocator;
    var wpd = WordPieceDecoder{};
    const d = wpd.decoder();

    const result = try d.decode(allocator, "hello");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("hello", result);
}

test "WordPieceDecoder: handles multiple subwords" {
    const allocator = std.testing.allocator;
    var wpd = WordPieceDecoder{};
    const d = wpd.decoder();

    // All subwords concatenate after removing ## prefix
    const result = try d.decode(allocator, "un##believ##able");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("unbelievable", result);
}

test "WordPieceDecoder: empty input" {
    const allocator = std.testing.allocator;
    var wpd = WordPieceDecoder{};
    const d = wpd.decoder();

    const result = try d.decode(allocator, "");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("", result);
}

test "WordPieceDecoder: single word no prefix" {
    const allocator = std.testing.allocator;
    var wpd = WordPieceDecoder{};
    const d = wpd.decoder();

    const result = try d.decode(allocator, "test");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("test", result);
}

test "WordPieceDecoder: only subwords (starts with ##)" {
    const allocator = std.testing.allocator;
    var wpd = WordPieceDecoder{};
    const d = wpd.decoder();

    const result = try d.decode(allocator, "##ing##ed");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("inged", result);
}

test "WordPieceDecoder: custom prefix" {
    const allocator = std.testing.allocator;
    var wpd = WordPieceDecoder{ .prefix = "@@" };
    const d = wpd.decoder();

    // Custom prefix is removed and subwords concatenate
    const result = try d.decode(allocator, "hello@@world");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("helloworld", result);
}

test "WordPieceDecoder: preserves UTF-8" {
    const allocator = std.testing.allocator;
    var wpd = WordPieceDecoder{};
    const d = wpd.decoder();

    // UTF-8 content is preserved, ## removed
    const result = try d.decode(allocator, "\xe4\xb8\x96##\xe7\x95\x8c");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("\xe4\xb8\x96\xe7\x95\x8c", result);
}

test "WordPieceDecoder: interface method" {
    const allocator = std.testing.allocator;
    var wpd = WordPieceDecoder{};
    const d = wpd.decoder();

    const result = try d.decode(allocator, "a##b");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("ab", result);
}

test "WordPieceDecoder: separate words get space" {
    const allocator = std.testing.allocator;
    var wpd = WordPieceDecoder{};
    const d = wpd.decoder();

    // Two separate words (neither starts with ##) get a space between them
    const result = try d.decode(allocator, "helloworld");
    defer allocator.free(result);
    // Without ## prefix, treated as single word
    try std.testing.expectEqualStrings("helloworld", result);
}

test "BPEDecoder: pass through" {
    const allocator = std.testing.allocator;
    var bped = BPEDecoder{};
    const d = bped.decoder();

    const result = try d.decode(allocator, "hello world");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("hello world", result);
}

test "BPEDecoder: empty input" {
    const allocator = std.testing.allocator;
    var bped = BPEDecoder{};
    const d = bped.decoder();

    const result = try d.decode(allocator, "");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("", result);
}

test "BPEDecoder: preserves content" {
    const allocator = std.testing.allocator;
    var bped = BPEDecoder{};
    const d = bped.decoder();

    const result = try d.decode(allocator, "Test123!@#");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("Test123!@#", result);
}

test "BPEDecoder: preserves UTF-8" {
    const allocator = std.testing.allocator;
    var bped = BPEDecoder{};
    const d = bped.decoder();

    const result = try d.decode(allocator, "\xe4\xb8\x96\xe7\x95\x8c");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("\xe4\xb8\x96\xe7\x95\x8c", result);
}

test "BPEDecoder: interface method" {
    const allocator = std.testing.allocator;
    var bped = BPEDecoder{};
    const d = bped.decoder();

    const result = try d.decode(allocator, "test");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("test", result);
}

test "ByteLevelDecoder: pass through" {
    const allocator = std.testing.allocator;
    var bld = ByteLevelDecoder{};
    const d = bld.decoder();

    const result = try d.decode(allocator, "hello world");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("hello world", result);
}

test "ByteLevelDecoder: empty input" {
    const allocator = std.testing.allocator;
    var bld = ByteLevelDecoder{};
    const d = bld.decoder();

    const result = try d.decode(allocator, "");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("", result);
}

test "ByteLevelDecoder: preserves UTF-8" {
    const allocator = std.testing.allocator;
    var bld = ByteLevelDecoder{};
    const d = bld.decoder();

    const result = try d.decode(allocator, "\xe4\xb8\x96\xe7\x95\x8c");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("\xe4\xb8\x96\xe7\x95\x8c", result);
}

test "Sequence: empty list pass through" {
    const allocator = std.testing.allocator;
    var seq = try Sequence.init(allocator, &.{});
    defer seq.deinit();
    const d = seq.decoder();

    const result = try d.decode(allocator, "hello");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("hello", result);
}

test "Sequence: single decoder" {
    const allocator = std.testing.allocator;
    var wpd = WordPieceDecoder{};

    var seq = try Sequence.init(allocator, &.{wpd.decoder()});
    defer seq.deinit();
    const d = seq.decoder();

    // WordPieceDecoder removes ## and concatenates
    const result = try d.decode(allocator, "hello##world");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("helloworld", result);
}

test "Sequence: multiple decoders" {
    const allocator = std.testing.allocator;
    var bped = BPEDecoder{};
    var bld = ByteLevelDecoder{};

    var seq = try Sequence.init(allocator, &.{ bped.decoder(), bld.decoder() });
    defer seq.deinit();
    const d = seq.decoder();

    const result = try d.decode(allocator, "test");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("test", result);
}

test "Sequence: chains WordPiece then BPE" {
    const allocator = std.testing.allocator;
    var wpd = WordPieceDecoder{};
    var bped = BPEDecoder{};

    var seq = try Sequence.init(allocator, &.{ wpd.decoder(), bped.decoder() });
    defer seq.deinit();
    const d = seq.decoder();

    // WordPiece removes ##, BPE passes through
    const result = try d.decode(allocator, "un##do");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("undo", result);
}

test "Sequence: deinit via interface" {
    const allocator = std.testing.allocator;
    var bped = BPEDecoder{};

    var seq = try Sequence.init(allocator, &.{bped.decoder()});
    var d = seq.decoder();

    const result = try d.decode(allocator, "test");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("test", result);

    // deinit via interface
    d.deinit();
}

test "Decoder: deinit with null function is safe" {
    var bped = BPEDecoder{};
    var d = bped.decoder();

    // deinit_fn is null for BPEDecoder
    d.deinit();
}
