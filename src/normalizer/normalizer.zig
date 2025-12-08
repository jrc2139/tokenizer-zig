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
