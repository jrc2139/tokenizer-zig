//! Vocabulary management for added tokens

const std = @import("std");
const types = @import("types.zig");
const AddedToken = types.AddedToken;

/// Vocabulary for managing added tokens (special and regular)
pub const Vocab = struct {
    allocator: std.mem.Allocator,
    token_to_id: std.StringHashMapUnmanaged(u32),
    id_to_token: std.AutoHashMapUnmanaged(u32, []const u8),
    special_tokens: std.StringHashMapUnmanaged(void),
    next_id: u32,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .token_to_id = std.StringHashMapUnmanaged(u32){},
            .id_to_token = std.AutoHashMapUnmanaged(u32, []const u8){},
            .special_tokens = std.StringHashMapUnmanaged(void){},
            .next_id = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        // Free all stored token strings
        var it = self.id_to_token.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.value_ptr.*);
        }
        self.token_to_id.deinit(self.allocator);
        self.id_to_token.deinit(self.allocator);
        self.special_tokens.deinit(self.allocator);
    }

    /// Add a special token, returns true if newly added
    pub fn addSpecialToken(self: *Self, token: AddedToken) !bool {
        if (self.token_to_id.get(token.content) != null) {
            return false;
        }

        const id = token.id orelse self.next_id;
        if (id >= self.next_id) {
            self.next_id = id + 1;
        }

        const content_copy = try self.allocator.dupe(u8, token.content);
        errdefer self.allocator.free(content_copy);

        try self.token_to_id.put(self.allocator, content_copy, id);
        try self.id_to_token.put(self.allocator, id, content_copy);
        try self.special_tokens.put(self.allocator, content_copy, {});

        return true;
    }

    /// Add a regular token, returns true if newly added
    pub fn addToken(self: *Self, token: AddedToken) !bool {
        if (self.token_to_id.get(token.content) != null) {
            return false;
        }

        const id = token.id orelse self.next_id;
        if (id >= self.next_id) {
            self.next_id = id + 1;
        }

        const content_copy = try self.allocator.dupe(u8, token.content);
        errdefer self.allocator.free(content_copy);

        try self.token_to_id.put(self.allocator, content_copy, id);
        try self.id_to_token.put(self.allocator, id, content_copy);

        if (token.special) {
            try self.special_tokens.put(self.allocator, content_copy, {});
        }

        return true;
    }

    /// Look up token by string
    pub fn tokenToId(self: *const Self, token: []const u8) ?u32 {
        return self.token_to_id.get(token);
    }

    /// Look up token by ID
    pub fn idToToken(self: *const Self, id: u32) ?[]const u8 {
        return self.id_to_token.get(id);
    }

    /// Check if token is special
    pub fn isSpecialToken(self: *const Self, token: []const u8) bool {
        return self.special_tokens.get(token) != null;
    }

    /// Get number of added tokens
    pub fn len(self: *const Self) usize {
        return self.token_to_id.count();
    }
};

test "vocab basic operations" {
    const allocator = std.testing.allocator;

    var vocab = Vocab.init(allocator);
    defer vocab.deinit();

    // Add special token
    const added = try vocab.addSpecialToken(AddedToken.init("[CLS]", true));
    try std.testing.expect(added);

    // Should be able to look it up
    const id = vocab.tokenToId("[CLS]");
    try std.testing.expect(id != null);

    const token = vocab.idToToken(id.?);
    try std.testing.expect(token != null);
    try std.testing.expectEqualStrings("[CLS]", token.?);

    // Should be marked as special
    try std.testing.expect(vocab.isSpecialToken("[CLS]"));
}
