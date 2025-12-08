//! Encoding represents the output of tokenization

const std = @import("std");
const lib = @import("lib.zig");
const Token = @import("token.zig").Token;

/// Encoding represents the complete output of tokenization
pub const Encoding = struct {
    allocator: std.mem.Allocator,
    ids: []u32,
    type_ids: []u32,
    tokens: [][]const u8,
    offsets: []lib.Offset,
    special_token_mask: []u32,
    attention_mask: []u32,
    words: ?[]i32, // -1 for none
    overflowing: []Encoding,
    owns_token_strs: bool = false, // Whether we own the token strings and need to free them

    const Self = @This();

    /// Create an encoding from tokens
    pub fn fromTokens(allocator: std.mem.Allocator, tokens: []const Token) !Self {
        const n = tokens.len;

        var ids = try allocator.alloc(u32, n);
        errdefer allocator.free(ids);

        var type_ids = try allocator.alloc(u32, n);
        errdefer allocator.free(type_ids);

        var token_strs = try allocator.alloc([]const u8, n);
        errdefer {
            for (token_strs[0..n]) |str| {
                if (str.len > 0) allocator.free(str);
            }
            allocator.free(token_strs);
        }

        var offsets = try allocator.alloc(lib.Offset, n);
        errdefer allocator.free(offsets);

        var special_token_mask = try allocator.alloc(u32, n);
        errdefer allocator.free(special_token_mask);

        var attention_mask = try allocator.alloc(u32, n);
        errdefer allocator.free(attention_mask);

        for (tokens, 0..) |tok, i| {
            ids[i] = tok.id;
            type_ids[i] = 0;
            // Copy the token string to own it
            token_strs[i] = try allocator.dupe(u8, tok.value);
            offsets[i] = tok.offset;
            special_token_mask[i] = 0;
            attention_mask[i] = 1;
        }

        return .{
            .allocator = allocator,
            .ids = ids,
            .type_ids = type_ids,
            .tokens = token_strs,
            .offsets = offsets,
            .special_token_mask = special_token_mask,
            .attention_mask = attention_mask,
            .words = null,
            .overflowing = &.{},
            .owns_token_strs = true,
        };
    }

    /// Create an empty encoding
    pub fn empty(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .ids = &.{},
            .type_ids = &.{},
            .tokens = &.{},
            .offsets = &.{},
            .special_token_mask = &.{},
            .attention_mask = &.{},
            .words = null,
            .overflowing = &.{},
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.ids.len > 0) {
            self.allocator.free(self.ids);
            self.allocator.free(self.type_ids);
            // Free token strings if we own them
            if (self.owns_token_strs) {
                for (self.tokens) |str| {
                    self.allocator.free(str);
                }
            }
            self.allocator.free(self.tokens);
            self.allocator.free(self.offsets);
            self.allocator.free(self.special_token_mask);
            self.allocator.free(self.attention_mask);
        }
        if (self.words) |w| {
            self.allocator.free(w);
        }
        for (self.overflowing) |*o| {
            o.deinit();
        }
        if (self.overflowing.len > 0) {
            self.allocator.free(self.overflowing);
        }
    }

    /// Get the number of tokens
    pub fn len(self: *const Self) usize {
        return self.ids.len;
    }

    /// Check if empty
    pub fn isEmpty(self: *const Self) bool {
        return self.ids.len == 0;
    }

    /// Get IDs
    pub fn getIds(self: *const Self) []const u32 {
        return self.ids;
    }

    /// Get tokens
    pub fn getTokens(self: *const Self) []const []const u8 {
        return self.tokens;
    }

    /// Get attention mask
    pub fn getAttentionMask(self: *const Self) []const u32 {
        return self.attention_mask;
    }

    /// Truncate the encoding to max_length
    pub fn truncate(self: *Self, max_length: usize, stride: usize) !void {
        if (self.ids.len <= max_length) {
            return;
        }

        _ = stride; // TODO: implement stride/overflowing

        // Simple truncation
        self.ids = self.ids[0..max_length];
        self.type_ids = self.type_ids[0..max_length];
        self.tokens = self.tokens[0..max_length];
        self.offsets = self.offsets[0..max_length];
        self.special_token_mask = self.special_token_mask[0..max_length];
        self.attention_mask = self.attention_mask[0..max_length];
        if (self.words) |w| {
            self.words = w[0..max_length];
        }
    }

    /// Pad the encoding to target_length
    pub fn pad(self: *Self, allocator: std.mem.Allocator, params: lib.PaddingParams) !void {
        const target_length = params.length orelse return; // batch_longest handled elsewhere

        if (self.ids.len >= target_length) {
            return;
        }

        const pad_len = target_length - self.ids.len;

        // Allocate new arrays
        var new_ids = try allocator.alloc(u32, target_length);
        var new_type_ids = try allocator.alloc(u32, target_length);
        var new_tokens = try allocator.alloc([]const u8, target_length);
        var new_offsets = try allocator.alloc(lib.Offset, target_length);
        var new_special = try allocator.alloc(u32, target_length);
        var new_attention = try allocator.alloc(u32, target_length);

        switch (params.direction) {
            .right => {
                @memcpy(new_ids[0..self.ids.len], self.ids);
                @memcpy(new_type_ids[0..self.type_ids.len], self.type_ids);
                @memcpy(new_tokens[0..self.tokens.len], self.tokens);
                @memcpy(new_offsets[0..self.offsets.len], self.offsets);
                @memcpy(new_special[0..self.special_token_mask.len], self.special_token_mask);
                @memcpy(new_attention[0..self.attention_mask.len], self.attention_mask);

                for (self.ids.len..target_length) |i| {
                    new_ids[i] = params.pad_id;
                    new_type_ids[i] = params.pad_type_id;
                    new_tokens[i] = params.pad_token;
                    new_offsets[i] = lib.Offset.init(0, 0);
                    new_special[i] = 1;
                    new_attention[i] = 0;
                }
            },
            .left => {
                for (0..pad_len) |i| {
                    new_ids[i] = params.pad_id;
                    new_type_ids[i] = params.pad_type_id;
                    new_tokens[i] = params.pad_token;
                    new_offsets[i] = lib.Offset.init(0, 0);
                    new_special[i] = 1;
                    new_attention[i] = 0;
                }

                @memcpy(new_ids[pad_len..], self.ids);
                @memcpy(new_type_ids[pad_len..], self.type_ids);
                @memcpy(new_tokens[pad_len..], self.tokens);
                @memcpy(new_offsets[pad_len..], self.offsets);
                @memcpy(new_special[pad_len..], self.special_token_mask);
                @memcpy(new_attention[pad_len..], self.attention_mask);
            },
        }

        // Free old arrays if they were allocated
        if (self.ids.len > 0) {
            allocator.free(self.ids);
            allocator.free(self.type_ids);
            allocator.free(self.tokens);
            allocator.free(self.offsets);
            allocator.free(self.special_token_mask);
            allocator.free(self.attention_mask);
        }

        self.ids = new_ids;
        self.type_ids = new_type_ids;
        self.tokens = new_tokens;
        self.offsets = new_offsets;
        self.special_token_mask = new_special;
        self.attention_mask = new_attention;
    }

    /// Merge with another encoding
    pub fn mergeWith(self: *Self, other: *const Self, growing_offsets: bool) !void {
        const new_len = self.ids.len + other.ids.len;

        var new_ids = try self.allocator.alloc(u32, new_len);
        @memcpy(new_ids[0..self.ids.len], self.ids);
        @memcpy(new_ids[self.ids.len..], other.ids);

        var new_type_ids = try self.allocator.alloc(u32, new_len);
        @memcpy(new_type_ids[0..self.type_ids.len], self.type_ids);
        @memcpy(new_type_ids[self.type_ids.len..], other.type_ids);

        var new_tokens = try self.allocator.alloc([]const u8, new_len);
        @memcpy(new_tokens[0..self.tokens.len], self.tokens);
        @memcpy(new_tokens[self.tokens.len..], other.tokens);

        var new_offsets = try self.allocator.alloc(lib.Offset, new_len);
        @memcpy(new_offsets[0..self.offsets.len], self.offsets);

        // Adjust offsets for growing_offsets
        var offset_adjustment: u32 = 0;
        if (growing_offsets and self.offsets.len > 0) {
            offset_adjustment = self.offsets[self.offsets.len - 1].end;
        }

        for (other.offsets, 0..) |off, i| {
            new_offsets[self.offsets.len + i] = lib.Offset.init(
                off.start + offset_adjustment,
                off.end + offset_adjustment,
            );
        }

        var new_special = try self.allocator.alloc(u32, new_len);
        @memcpy(new_special[0..self.special_token_mask.len], self.special_token_mask);
        @memcpy(new_special[self.special_token_mask.len..], other.special_token_mask);

        var new_attention = try self.allocator.alloc(u32, new_len);
        @memcpy(new_attention[0..self.attention_mask.len], self.attention_mask);
        @memcpy(new_attention[self.attention_mask.len..], other.attention_mask);

        // Free old and assign new
        if (self.ids.len > 0) {
            self.allocator.free(self.ids);
            self.allocator.free(self.type_ids);
            self.allocator.free(self.tokens);
            self.allocator.free(self.offsets);
            self.allocator.free(self.special_token_mask);
            self.allocator.free(self.attention_mask);
        }

        self.ids = new_ids;
        self.type_ids = new_type_ids;
        self.tokens = new_tokens;
        self.offsets = new_offsets;
        self.special_token_mask = new_special;
        self.attention_mask = new_attention;
    }
};
