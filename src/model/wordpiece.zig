//! WordPiece tokenization model
//!
//! WordPiece is used by BERT and similar models. It splits words into
//! subword units, with non-first subwords prefixed with "##".

const std = @import("std");
const model = @import("model.zig");
const Token = @import("../token.zig").Token;

/// WordPiece tokenization model
pub const WordPiece = struct {
    allocator: std.mem.Allocator,
    vocab: std.StringHashMapUnmanaged(u32),
    vocab_r: std.AutoHashMapUnmanaged(u32, []const u8),
    unk_token: []const u8,
    continuing_subword_prefix: []const u8,
    max_input_chars_per_word: usize,

    const Self = @This();

    pub const Config = struct {
        unk_token: []const u8 = "[UNK]",
        continuing_subword_prefix: []const u8 = "##",
        max_input_chars_per_word: usize = 100,
    };

    pub fn init(allocator: std.mem.Allocator, vocab: std.StringHashMapUnmanaged(u32), config: Config) !Self {
        // Build reverse vocab
        var vocab_r = std.AutoHashMapUnmanaged(u32, []const u8){};
        var it = vocab.iterator();
        while (it.next()) |entry| {
            try vocab_r.put(allocator, entry.value_ptr.*, entry.key_ptr.*);
        }

        return .{
            .allocator = allocator,
            .vocab = vocab,
            .vocab_r = vocab_r,
            .unk_token = config.unk_token,
            .continuing_subword_prefix = config.continuing_subword_prefix,
            .max_input_chars_per_word = config.max_input_chars_per_word,
        };
    }

    pub fn deinit(self: *Self) void {
        // Free vocab keys (strings)
        var it = self.vocab.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.vocab.deinit(self.allocator);
        self.vocab_r.deinit(self.allocator);
    }

    /// Get as generic Model interface
    pub fn getModel(self: *Self) model.Model {
        return .{
            .ptr = self,
            .tokenize_fn = tokenizeImpl,
            .token_to_id_fn = tokenToIdImpl,
            .id_to_token_fn = idToTokenImpl,
            .get_vocab_size_fn = getVocabSizeImpl,
            .deinit = deinitImpl,
            .destroy = destroyImpl,
        };
    }

    fn tokenizeImpl(ctx: *anyopaque, allocator: std.mem.Allocator, sequence: []const u8) anyerror![]Token {
        const self: *Self = @ptrCast(@alignCast(ctx));
        return self.tokenize(allocator, sequence);
    }

    fn tokenToIdImpl(ctx: *anyopaque, token: []const u8) ?u32 {
        const self: *Self = @ptrCast(@alignCast(ctx));
        return self.vocab.get(token);
    }

    fn idToTokenImpl(ctx: *anyopaque, id: u32) ?[]const u8 {
        const self: *Self = @ptrCast(@alignCast(ctx));
        return self.vocab_r.get(id);
    }

    fn getVocabSizeImpl(ctx: *anyopaque) usize {
        const self: *Self = @ptrCast(@alignCast(ctx));
        return self.vocab.count();
    }

    fn deinitImpl(ctx: *anyopaque) void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        self.deinit();
    }

    fn destroyImpl(allocator: std.mem.Allocator, ctx: *anyopaque) void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        allocator.destroy(self);
    }

    /// Tokenize a single word using WordPiece algorithm
    pub fn tokenize(self: *Self, allocator: std.mem.Allocator, sequence: []const u8) ![]Token {
        var tokens = std.ArrayListUnmanaged(Token){};
        errdefer tokens.deinit(allocator);

        const chars = sequence;
        const char_len = chars.len;

        // If word is too long, return UNK
        if (char_len > self.max_input_chars_per_word) {
            const unk_id = self.vocab.get(self.unk_token) orelse return error.MissingUnkToken;
            try tokens.append(allocator, Token.init(
                unk_id,
                self.unk_token,
                0,
                @intCast(char_len),
            ));
            return try tokens.toOwnedSlice(allocator);
        }

        var is_bad = false;
        var start: usize = 0;

        while (start < char_len) {
            var end = char_len;
            var cur_substr: ?[]const u8 = null;
            var cur_id: ?u32 = null;

            while (start < end) {
                var substr_buf: [512]u8 = undefined;
                var substr: []const u8 = undefined;

                if (start > 0) {
                    // Add continuing subword prefix
                    const prefix_len = self.continuing_subword_prefix.len;
                    const word_len = end - start;
                    if (prefix_len + word_len > substr_buf.len) {
                        end -= 1;
                        continue;
                    }
                    @memcpy(substr_buf[0..prefix_len], self.continuing_subword_prefix);
                    @memcpy(substr_buf[prefix_len .. prefix_len + word_len], chars[start..end]);
                    substr = substr_buf[0 .. prefix_len + word_len];
                } else {
                    substr = chars[start..end];
                }

                if (self.vocab.get(substr)) |id| {
                    cur_substr = substr;
                    cur_id = id;
                    break;
                }
                end -= 1;
            }

            if (cur_substr == null) {
                is_bad = true;
                break;
            }

            try tokens.append(allocator, Token.init(
                cur_id.?,
                cur_substr.?,
                @intCast(start),
                @intCast(end),
            ));
            start = end;
        }

        if (is_bad) {
            // Clear tokens and return UNK
            tokens.clearAndFree(allocator);
            const unk_id = self.vocab.get(self.unk_token) orelse return error.MissingUnkToken;
            try tokens.append(allocator, Token.init(
                unk_id,
                self.unk_token,
                0,
                @intCast(char_len),
            ));
        }

        return try tokens.toOwnedSlice(allocator);
    }
};
