//! Model interface for tokenization models (BPE, WordPiece, etc.)

const std = @import("std");
const Token = @import("../token.zig").Token;

/// Function pointer type for tokenization
pub const TokenizeFn = *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, sequence: []const u8) anyerror![]Token;
pub const TokenToIdFn = *const fn (ctx: *anyopaque, token: []const u8) ?u32;
pub const IdToTokenFn = *const fn (ctx: *anyopaque, id: u32) ?[]const u8;
pub const GetVocabSizeFn = *const fn (ctx: *anyopaque) usize;
pub const DeinitFn = *const fn (ctx: *anyopaque) void;
pub const DestroyFn = *const fn (allocator: std.mem.Allocator, ctx: *anyopaque) void;

/// Model interface - implemented by BPE, WordPiece, etc.
pub const Model = struct {
    ptr: *anyopaque,
    tokenize_fn: TokenizeFn,
    token_to_id_fn: TokenToIdFn,
    id_to_token_fn: IdToTokenFn,
    get_vocab_size_fn: GetVocabSizeFn,
    deinit: ?DeinitFn = null,
    destroy: ?DestroyFn = null, // Frees the struct itself

    const Self = @This();

    pub fn tokenize(self: Self, allocator: std.mem.Allocator, sequence: []const u8) ![]Token {
        return self.tokenize_fn(self.ptr, allocator, sequence);
    }

    pub fn tokenToId(self: Self, token: []const u8) ?u32 {
        return self.token_to_id_fn(self.ptr, token);
    }

    pub fn idToToken(self: Self, id: u32) ?[]const u8 {
        return self.id_to_token_fn(self.ptr, id);
    }

    pub fn getVocabSize(self: Self) usize {
        return self.get_vocab_size_fn(self.ptr);
    }
};

// Re-export model implementations
pub const WordPiece = @import("wordpiece.zig").WordPiece;
pub const BPE = @import("bpe.zig").BPE;
