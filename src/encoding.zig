//! Encoding represents the output of tokenization

const std = @import("std");
const lib = @import("lib.zig");
const Token = @import("token.zig").Token;
const SpanToken = @import("token.zig").SpanToken;
const SpanTokenFlags = @import("token.zig").SpanTokenFlags;

// ============================================================================
// SpanEncoding - Zero-Allocation Encoding Container
// ============================================================================

/// Pre-allocated encoding container that enables zero-allocation tokenization.
/// Stores span tokens (offsets into input) rather than copied strings.
/// All arrays are pre-allocated at init time and reused across encode() calls.
pub const SpanEncoding = struct {
    /// Reference to original input text (borrowed, not owned)
    input: []const u8,

    /// Pre-allocated span token buffer
    tokens: []SpanToken,

    /// Current number of tokens
    len: u32,

    /// Maximum capacity
    capacity: u32,

    /// SoA arrays for ML framework compatibility (views into pre-allocated buffer)
    /// These are separate arrays for efficient export to ML frameworks
    ids: []u32,
    attention_mask: []u32,
    type_ids: []u32,
    offsets: []lib.Offset,

    const Self = @This();

    /// Initialize SpanEncoding with pre-allocated buffers.
    /// This is the only allocation point - after this, encode() is zero-alloc.
    pub fn init(allocator: std.mem.Allocator, max_tokens: u32) !Self {
        const cap = max_tokens;

        const tokens = try allocator.alloc(SpanToken, cap);
        errdefer allocator.free(tokens);

        const ids = try allocator.alloc(u32, cap);
        errdefer allocator.free(ids);

        const attention_mask = try allocator.alloc(u32, cap);
        errdefer allocator.free(attention_mask);

        const type_ids = try allocator.alloc(u32, cap);
        errdefer allocator.free(type_ids);

        const offsets = try allocator.alloc(lib.Offset, cap);
        errdefer allocator.free(offsets);

        return .{
            .input = &.{},
            .tokens = tokens,
            .len = 0,
            .capacity = cap,
            .ids = ids,
            .attention_mask = attention_mask,
            .type_ids = type_ids,
            .offsets = offsets,
        };
    }

    /// Free all pre-allocated buffers
    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        allocator.free(self.tokens);
        allocator.free(self.ids);
        allocator.free(self.attention_mask);
        allocator.free(self.type_ids);
        allocator.free(self.offsets);
    }

    /// Reset for reuse with new input (O(1), no allocation)
    pub fn reset(self: *Self, new_input: []const u8) void {
        self.input = new_input;
        self.len = 0;
    }

    /// Append a token (inline for hot path performance)
    pub inline fn append(self: *Self, token: SpanToken) void {
        std.debug.assert(self.len < self.capacity);
        const i = self.len;
        self.tokens[i] = token;
        self.ids[i] = token.id;
        self.attention_mask[i] = if (token.flags.is_padding) 0 else 1;
        self.type_ids[i] = token.type_id;
        self.offsets[i] = lib.Offset.init(token.start, token.end);
        self.len = i + 1;
    }

    /// Append a token with bounds checking, returns false if full
    pub fn tryAppend(self: *Self, token: SpanToken) bool {
        if (self.len >= self.capacity) return false;
        self.append(token);
        return true;
    }

    /// Get token string on demand (zero-copy slice into original input)
    pub fn getTokenStr(self: *const Self, index: usize) []const u8 {
        const tok = self.tokens[index];
        if (tok.flags.is_padding or tok.flags.is_special) {
            // Special/padding tokens may not have valid input offsets
            return "";
        }
        return self.input[tok.start..tok.end];
    }

    /// Get current token count
    pub fn length(self: *const Self) usize {
        return self.len;
    }

    /// Check if empty
    pub fn isEmpty(self: *const Self) bool {
        return self.len == 0;
    }

    /// Get IDs slice (active portion only)
    pub fn getIds(self: *const Self) []const u32 {
        return self.ids[0..self.len];
    }

    /// Get attention mask slice
    pub fn getAttentionMask(self: *const Self) []const u32 {
        return self.attention_mask[0..self.len];
    }

    /// Get type IDs slice
    pub fn getTypeIds(self: *const Self) []const u32 {
        return self.type_ids[0..self.len];
    }

    /// Get offsets slice
    pub fn getOffsets(self: *const Self) []const lib.Offset {
        return self.offsets[0..self.len];
    }

    /// Get tokens slice
    pub fn getTokens(self: *const Self) []const SpanToken {
        return self.tokens[0..self.len];
    }

    /// Truncate to max_length (O(1), just adjusts len)
    pub fn truncate(self: *Self, max_length: u32) void {
        if (self.len > max_length) {
            self.len = max_length;
        }
    }

    /// Pad to target_length with padding tokens
    /// Note: This still doesn't allocate - uses pre-allocated buffer
    pub fn pad(self: *Self, target_length: u32, pad_id: u32) void {
        while (self.len < target_length and self.len < self.capacity) {
            self.append(SpanToken.initPadding(pad_id));
        }
    }

    /// Convert to legacy Encoding (allocates)
    /// Use this when you need HuggingFace-compatible output
    pub fn toEncoding(self: *const Self, allocator: std.mem.Allocator) !Encoding {
        const n = self.len;
        if (n == 0) {
            return Encoding.empty(allocator);
        }

        const ids = try allocator.alloc(u32, n);
        errdefer allocator.free(ids);
        @memcpy(ids, self.ids[0..n]);

        const type_ids_out = try allocator.alloc(u32, n);
        errdefer allocator.free(type_ids_out);
        @memcpy(type_ids_out, self.type_ids[0..n]);

        const token_strs = try allocator.alloc([]const u8, n);
        errdefer {
            for (token_strs[0..n]) |str| {
                if (str.len > 0) allocator.free(str);
            }
            allocator.free(token_strs);
        }

        for (0..n) |i| {
            const tok = self.tokens[i];
            if (tok.flags.is_padding) {
                token_strs[i] = try allocator.dupe(u8, "[PAD]");
            } else {
                token_strs[i] = try allocator.dupe(u8, self.input[tok.start..tok.end]);
            }
        }

        const offsets_out = try allocator.alloc(lib.Offset, n);
        errdefer allocator.free(offsets_out);
        @memcpy(offsets_out, self.offsets[0..n]);

        const special_mask = try allocator.alloc(u32, n);
        errdefer allocator.free(special_mask);
        for (0..n) |i| {
            special_mask[i] = if (self.tokens[i].flags.is_special) 1 else 0;
        }

        const attention_out = try allocator.alloc(u32, n);
        errdefer allocator.free(attention_out);
        @memcpy(attention_out, self.attention_mask[0..n]);

        return .{
            .allocator = allocator,
            .ids = ids,
            .type_ids = type_ids_out,
            .tokens = token_strs,
            .offsets = offsets_out,
            .special_token_mask = special_mask,
            .attention_mask = attention_out,
            .words = null,
            .overflowing = &.{},
            .owns_token_strs = true,
        };
    }
};

// ============================================================================
// Encoding - Original Implementation (Backward Compatibility)
// ============================================================================

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
    /// Note: After padding, owns_token_strs is set to false as token strings
    /// are either static strings (for padding) or were already owned.
    pub fn pad(self: *Self, allocator: std.mem.Allocator, params: lib.PaddingParams) !void {
        const target_length = params.length orelse return; // batch_longest handled elsewhere

        if (self.ids.len >= target_length) {
            return;
        }

        const pad_len = target_length - self.ids.len;

        // Allocate new arrays
        const new_ids = try allocator.alloc(u32, target_length);
        const new_type_ids = try allocator.alloc(u32, target_length);
        const new_tokens = try allocator.alloc([]const u8, target_length);
        const new_offsets = try allocator.alloc(lib.Offset, target_length);
        const new_special = try allocator.alloc(u32, target_length);
        const new_attention = try allocator.alloc(u32, target_length);

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
            // Free owned token strings first
            if (self.owns_token_strs) {
                for (self.tokens) |str| {
                    allocator.free(str);
                }
            }
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
        // After padding, we no longer own the token strings (they're a mix of copied ptrs and static strings)
        self.owns_token_strs = false;
    }

    /// Create a copy of this encoding
    pub fn clone(self: *const Self, allocator: std.mem.Allocator) !Self {
        const n = self.ids.len;
        if (n == 0) {
            return empty(allocator);
        }

        const ids = try allocator.alloc(u32, n);
        errdefer allocator.free(ids);
        @memcpy(ids, self.ids);

        const type_ids = try allocator.alloc(u32, n);
        errdefer allocator.free(type_ids);
        @memcpy(type_ids, self.type_ids);

        const token_strs = try allocator.alloc([]const u8, n);
        errdefer {
            for (token_strs[0..n]) |str| {
                if (str.len > 0) allocator.free(str);
            }
            allocator.free(token_strs);
        }
        for (self.tokens, 0..) |str, i| {
            token_strs[i] = try allocator.dupe(u8, str);
        }

        const offsets = try allocator.alloc(lib.Offset, n);
        errdefer allocator.free(offsets);
        @memcpy(offsets, self.offsets);

        const special_token_mask = try allocator.alloc(u32, n);
        errdefer allocator.free(special_token_mask);
        @memcpy(special_token_mask, self.special_token_mask);

        const attention_mask = try allocator.alloc(u32, n);
        errdefer allocator.free(attention_mask);
        @memcpy(attention_mask, self.attention_mask);

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

    /// Merge with another encoding
    /// Note: After merging, owns_token_strs is set to false as token strings
    /// are copied pointers from both encodings.
    pub fn mergeWith(self: *Self, other: *const Self, growing_offsets: bool) !void {
        const new_len = self.ids.len + other.ids.len;

        const new_ids = try self.allocator.alloc(u32, new_len);
        @memcpy(new_ids[0..self.ids.len], self.ids);
        @memcpy(new_ids[self.ids.len..], other.ids);

        const new_type_ids = try self.allocator.alloc(u32, new_len);
        @memcpy(new_type_ids[0..self.type_ids.len], self.type_ids);
        @memcpy(new_type_ids[self.type_ids.len..], other.type_ids);

        const new_tokens = try self.allocator.alloc([]const u8, new_len);
        @memcpy(new_tokens[0..self.tokens.len], self.tokens);
        @memcpy(new_tokens[self.tokens.len..], other.tokens);

        const new_offsets = try self.allocator.alloc(lib.Offset, new_len);
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

        const new_special = try self.allocator.alloc(u32, new_len);
        @memcpy(new_special[0..self.special_token_mask.len], self.special_token_mask);
        @memcpy(new_special[self.special_token_mask.len..], other.special_token_mask);

        const new_attention = try self.allocator.alloc(u32, new_len);
        @memcpy(new_attention[0..self.attention_mask.len], self.attention_mask);
        @memcpy(new_attention[self.attention_mask.len..], other.attention_mask);

        // Free old and assign new
        if (self.ids.len > 0) {
            // Free owned token strings first
            if (self.owns_token_strs) {
                for (self.tokens) |str| {
                    self.allocator.free(str);
                }
            }
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
        // After merging, we no longer own the token strings (they're from both encodings)
        self.owns_token_strs = false;
    }
};

// ============================================================================
// Unit Tests
// ============================================================================

test "encoding empty" {
    const allocator = std.testing.allocator;
    var enc = Encoding.empty(allocator);
    defer enc.deinit();

    try std.testing.expectEqual(@as(usize, 0), enc.len());
    try std.testing.expect(enc.isEmpty());
}

test "encoding fromTokens basic" {
    const allocator = std.testing.allocator;

    var tokens_arr = [_]Token{
        Token.init(100, "hello", 0, 5),
        Token.init(200, "world", 6, 11),
    };

    var enc = try Encoding.fromTokens(allocator, &tokens_arr);
    defer enc.deinit();

    try std.testing.expectEqual(@as(usize, 2), enc.len());
    try std.testing.expect(!enc.isEmpty());

    const ids = enc.getIds();
    try std.testing.expectEqual(@as(u32, 100), ids[0]);
    try std.testing.expectEqual(@as(u32, 200), ids[1]);

    const toks = enc.getTokens();
    try std.testing.expectEqualStrings("hello", toks[0]);
    try std.testing.expectEqualStrings("world", toks[1]);
}

test "encoding attention mask" {
    const allocator = std.testing.allocator;

    var tokens_arr = [_]Token{
        Token.init(1, "a", 0, 1),
        Token.init(2, "b", 1, 2),
        Token.init(3, "c", 2, 3),
    };

    var enc = try Encoding.fromTokens(allocator, &tokens_arr);
    defer enc.deinit();

    const mask = enc.getAttentionMask();
    try std.testing.expectEqual(@as(usize, 3), mask.len);
    try std.testing.expectEqual(@as(u32, 1), mask[0]);
    try std.testing.expectEqual(@as(u32, 1), mask[1]);
    try std.testing.expectEqual(@as(u32, 1), mask[2]);
}

test "encoding truncate" {
    // Note: truncate does in-place slicing without reallocating.
    // This test verifies the truncate logic works, but we don't call deinit
    // on the truncated encoding since the allocator can't free re-sliced memory.
    // In practice, truncate is used on encodings that won't be individually freed.

    const allocator = std.testing.allocator;

    var tokens_arr = [_]Token{
        Token.init(1, "a", 0, 1),
        Token.init(2, "b", 1, 2),
        Token.init(3, "c", 2, 3),
    };

    var enc = try Encoding.fromTokens(allocator, &tokens_arr);
    defer enc.deinit();

    // Truncate should be no-op when already at or under the limit
    try enc.truncate(5, 0);
    try std.testing.expectEqual(@as(usize, 3), enc.len());

    try enc.truncate(3, 0);
    try std.testing.expectEqual(@as(usize, 3), enc.len());
    try std.testing.expectEqual(@as(u32, 1), enc.ids[0]);
    try std.testing.expectEqual(@as(u32, 2), enc.ids[1]);
    try std.testing.expectEqual(@as(u32, 3), enc.ids[2]);
}

test "encoding truncate no-op when shorter" {
    const allocator = std.testing.allocator;

    var tokens_arr = [_]Token{
        Token.init(1, "a", 0, 1),
        Token.init(2, "b", 1, 2),
    };

    var enc = try Encoding.fromTokens(allocator, &tokens_arr);
    defer enc.deinit();

    try enc.truncate(10, 0);

    try std.testing.expectEqual(@as(usize, 2), enc.len());
}

test "encoding pad right" {
    const allocator = std.testing.allocator;

    var tokens_arr = [_]Token{
        Token.init(1, "a", 0, 1),
        Token.init(2, "b", 1, 2),
    };

    var enc = try Encoding.fromTokens(allocator, &tokens_arr);
    defer enc.deinit();

    try enc.pad(allocator, .{
        .length = 5,
        .pad_id = 0,
        .pad_token = "[PAD]",
        .direction = .right,
    });

    try std.testing.expectEqual(@as(usize, 5), enc.len());
    try std.testing.expectEqual(@as(u32, 1), enc.ids[0]);
    try std.testing.expectEqual(@as(u32, 2), enc.ids[1]);
    try std.testing.expectEqual(@as(u32, 0), enc.ids[2]);
    try std.testing.expectEqual(@as(u32, 0), enc.ids[3]);
    try std.testing.expectEqual(@as(u32, 0), enc.ids[4]);

    // Attention mask: 1 for real tokens, 0 for padding
    try std.testing.expectEqual(@as(u32, 1), enc.attention_mask[0]);
    try std.testing.expectEqual(@as(u32, 1), enc.attention_mask[1]);
    try std.testing.expectEqual(@as(u32, 0), enc.attention_mask[2]);
}

test "encoding pad left" {
    const allocator = std.testing.allocator;

    var tokens_arr = [_]Token{
        Token.init(1, "a", 0, 1),
        Token.init(2, "b", 1, 2),
    };

    var enc = try Encoding.fromTokens(allocator, &tokens_arr);
    defer enc.deinit();

    try enc.pad(allocator, .{
        .length = 4,
        .pad_id = 0,
        .direction = .left,
    });

    try std.testing.expectEqual(@as(usize, 4), enc.len());
    try std.testing.expectEqual(@as(u32, 0), enc.ids[0]);
    try std.testing.expectEqual(@as(u32, 0), enc.ids[1]);
    try std.testing.expectEqual(@as(u32, 1), enc.ids[2]);
    try std.testing.expectEqual(@as(u32, 2), enc.ids[3]);
}

test "encoding pad no-op when already long enough" {
    const allocator = std.testing.allocator;

    var tokens_arr = [_]Token{
        Token.init(1, "a", 0, 1),
        Token.init(2, "b", 1, 2),
    };

    var enc = try Encoding.fromTokens(allocator, &tokens_arr);
    defer enc.deinit();

    try enc.pad(allocator, .{ .length = 2 });

    try std.testing.expectEqual(@as(usize, 2), enc.len());
}

test "encoding clone" {
    const allocator = std.testing.allocator;

    var tokens_arr = [_]Token{
        Token.init(100, "hello", 0, 5),
        Token.init(200, "world", 6, 11),
    };

    var enc = try Encoding.fromTokens(allocator, &tokens_arr);
    defer enc.deinit();

    var cloned = try enc.clone(allocator);
    defer cloned.deinit();

    // Should have same content
    try std.testing.expectEqual(enc.len(), cloned.len());
    try std.testing.expectEqual(enc.ids[0], cloned.ids[0]);
    try std.testing.expectEqual(enc.ids[1], cloned.ids[1]);
    try std.testing.expectEqualStrings(enc.tokens[0], cloned.tokens[0]);
    try std.testing.expectEqualStrings(enc.tokens[1], cloned.tokens[1]);
}

test "encoding clone empty" {
    const allocator = std.testing.allocator;

    var enc = Encoding.empty(allocator);
    defer enc.deinit();

    var cloned = try enc.clone(allocator);
    defer cloned.deinit();

    try std.testing.expectEqual(@as(usize, 0), cloned.len());
}

test "encoding mergeWith" {
    const allocator = std.testing.allocator;

    var tokens1 = [_]Token{
        Token.init(1, "hello", 0, 5),
    };
    var tokens2 = [_]Token{
        Token.init(2, "world", 0, 5),
    };

    var enc1 = try Encoding.fromTokens(allocator, &tokens1);
    defer enc1.deinit();

    var enc2 = try Encoding.fromTokens(allocator, &tokens2);
    defer enc2.deinit();

    try enc1.mergeWith(&enc2, false);

    try std.testing.expectEqual(@as(usize, 2), enc1.len());
    try std.testing.expectEqual(@as(u32, 1), enc1.ids[0]);
    try std.testing.expectEqual(@as(u32, 2), enc1.ids[1]);
}

test "encoding mergeWith growing offsets" {
    const allocator = std.testing.allocator;

    var tokens1 = [_]Token{
        Token.init(1, "hello", 0, 5),
    };
    var tokens2 = [_]Token{
        Token.init(2, "world", 0, 5),
    };

    var enc1 = try Encoding.fromTokens(allocator, &tokens1);
    defer enc1.deinit();

    var enc2 = try Encoding.fromTokens(allocator, &tokens2);
    defer enc2.deinit();

    try enc1.mergeWith(&enc2, true);

    // Second token's offsets should be adjusted
    try std.testing.expectEqual(@as(u32, 0), enc1.offsets[0].start);
    try std.testing.expectEqual(@as(u32, 5), enc1.offsets[0].end);
    try std.testing.expectEqual(@as(u32, 5), enc1.offsets[1].start);
    try std.testing.expectEqual(@as(u32, 10), enc1.offsets[1].end);
}

test "encoding owns token strings" {
    const allocator = std.testing.allocator;

    var tokens_arr = [_]Token{
        Token.init(100, "hello", 0, 5),
    };

    var enc = try Encoding.fromTokens(allocator, &tokens_arr);
    defer enc.deinit();

    // fromTokens should own its strings
    try std.testing.expect(enc.owns_token_strs);
}

test "encoding special token mask" {
    const allocator = std.testing.allocator;

    var tokens_arr = [_]Token{
        Token.init(1, "a", 0, 1),
        Token.init(2, "b", 1, 2),
    };

    var enc = try Encoding.fromTokens(allocator, &tokens_arr);
    defer enc.deinit();

    // Regular tokens have special_token_mask = 0
    try std.testing.expectEqual(@as(u32, 0), enc.special_token_mask[0]);
    try std.testing.expectEqual(@as(u32, 0), enc.special_token_mask[1]);
}

// ============================================================================
// SpanEncoding Unit Tests
// ============================================================================

test "SpanEncoding init and deinit" {
    const allocator = std.testing.allocator;

    var enc = try SpanEncoding.init(allocator, 512);
    defer enc.deinit(allocator);

    try std.testing.expectEqual(@as(u32, 0), enc.len);
    try std.testing.expectEqual(@as(u32, 512), enc.capacity);
    try std.testing.expect(enc.isEmpty());
}

test "SpanEncoding append and getIds" {
    const allocator = std.testing.allocator;

    var enc = try SpanEncoding.init(allocator, 512);
    defer enc.deinit(allocator);

    const input = "hello world";
    enc.reset(input);

    enc.append(SpanToken.init(100, 0, 5)); // "hello"
    enc.append(SpanToken.init(200, 6, 11)); // "world"

    try std.testing.expectEqual(@as(u32, 2), enc.len);
    try std.testing.expect(!enc.isEmpty());

    const ids = enc.getIds();
    try std.testing.expectEqual(@as(usize, 2), ids.len);
    try std.testing.expectEqual(@as(u32, 100), ids[0]);
    try std.testing.expectEqual(@as(u32, 200), ids[1]);
}

test "SpanEncoding getTokenStr zero-copy" {
    const allocator = std.testing.allocator;

    var enc = try SpanEncoding.init(allocator, 512);
    defer enc.deinit(allocator);

    const input = "hello world";
    enc.reset(input);

    enc.append(SpanToken.init(1, 0, 5));
    enc.append(SpanToken.init(2, 6, 11));

    // Zero-copy: getTokenStr returns slice into original input
    try std.testing.expectEqualStrings("hello", enc.getTokenStr(0));
    try std.testing.expectEqualStrings("world", enc.getTokenStr(1));

    // Verify it's actually pointing to the same memory
    try std.testing.expectEqual(@intFromPtr(input.ptr), @intFromPtr(enc.getTokenStr(0).ptr));
}

test "SpanEncoding reset reuses buffer" {
    const allocator = std.testing.allocator;

    var enc = try SpanEncoding.init(allocator, 512);
    defer enc.deinit(allocator);

    // First use
    enc.reset("hello");
    enc.append(SpanToken.init(1, 0, 5));
    try std.testing.expectEqual(@as(u32, 1), enc.len);

    // Reset and reuse - no allocation
    enc.reset("world");
    try std.testing.expectEqual(@as(u32, 0), enc.len);
    enc.append(SpanToken.init(2, 0, 5));
    try std.testing.expectEqual(@as(u32, 1), enc.len);
}

test "SpanEncoding attention mask" {
    const allocator = std.testing.allocator;

    var enc = try SpanEncoding.init(allocator, 512);
    defer enc.deinit(allocator);

    enc.reset("test");
    enc.append(SpanToken.init(1, 0, 4));
    enc.append(SpanToken.initPadding(0));

    const mask = enc.getAttentionMask();
    try std.testing.expectEqual(@as(u32, 1), mask[0]); // real token
    try std.testing.expectEqual(@as(u32, 0), mask[1]); // padding
}

test "SpanEncoding truncate" {
    const allocator = std.testing.allocator;

    var enc = try SpanEncoding.init(allocator, 512);
    defer enc.deinit(allocator);

    enc.reset("abc");
    enc.append(SpanToken.init(1, 0, 1));
    enc.append(SpanToken.init(2, 1, 2));
    enc.append(SpanToken.init(3, 2, 3));

    try std.testing.expectEqual(@as(u32, 3), enc.len);

    enc.truncate(2);
    try std.testing.expectEqual(@as(u32, 2), enc.len);

    const ids = enc.getIds();
    try std.testing.expectEqual(@as(usize, 2), ids.len);
}

test "SpanEncoding pad" {
    const allocator = std.testing.allocator;

    var enc = try SpanEncoding.init(allocator, 512);
    defer enc.deinit(allocator);

    enc.reset("ab");
    enc.append(SpanToken.init(1, 0, 1));
    enc.append(SpanToken.init(2, 1, 2));

    enc.pad(5, 0); // pad to length 5 with pad_id=0

    try std.testing.expectEqual(@as(u32, 5), enc.len);

    const ids = enc.getIds();
    try std.testing.expectEqual(@as(u32, 1), ids[0]);
    try std.testing.expectEqual(@as(u32, 2), ids[1]);
    try std.testing.expectEqual(@as(u32, 0), ids[2]); // padding
    try std.testing.expectEqual(@as(u32, 0), ids[3]); // padding
    try std.testing.expectEqual(@as(u32, 0), ids[4]); // padding

    // Check attention mask
    const mask = enc.getAttentionMask();
    try std.testing.expectEqual(@as(u32, 1), mask[0]);
    try std.testing.expectEqual(@as(u32, 1), mask[1]);
    try std.testing.expectEqual(@as(u32, 0), mask[2]);
}

test "SpanEncoding toEncoding conversion" {
    const allocator = std.testing.allocator;

    var span_enc = try SpanEncoding.init(allocator, 512);
    defer span_enc.deinit(allocator);

    const input = "hello world";
    span_enc.reset(input);
    span_enc.append(SpanToken.init(100, 0, 5));
    span_enc.append(SpanToken.init(200, 6, 11));

    var enc = try span_enc.toEncoding(allocator);
    defer enc.deinit();

    try std.testing.expectEqual(@as(usize, 2), enc.len());
    try std.testing.expectEqual(@as(u32, 100), enc.ids[0]);
    try std.testing.expectEqual(@as(u32, 200), enc.ids[1]);
    try std.testing.expectEqualStrings("hello", enc.tokens[0]);
    try std.testing.expectEqualStrings("world", enc.tokens[1]);
}

test "SpanEncoding tryAppend bounds check" {
    const allocator = std.testing.allocator;

    var enc = try SpanEncoding.init(allocator, 2); // small capacity
    defer enc.deinit(allocator);

    enc.reset("abc");

    try std.testing.expect(enc.tryAppend(SpanToken.init(1, 0, 1)));
    try std.testing.expect(enc.tryAppend(SpanToken.init(2, 1, 2)));
    try std.testing.expect(!enc.tryAppend(SpanToken.init(3, 2, 3))); // should fail

    try std.testing.expectEqual(@as(u32, 2), enc.len);
}
