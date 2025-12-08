//! tokenizer-zig: Pure Zig implementation of HuggingFace tokenizers
//!
//! This library provides tokenization functionality compatible with
//! HuggingFace tokenizer.json files, supporting WordPiece (BERT) and
//! BPE (GPT-2, RoBERTa) tokenization models.

const std = @import("std");
pub const Token = @import("token.zig").Token;
pub const Encoding = @import("encoding.zig").Encoding;
pub const Vocab = @import("vocab.zig").Vocab;
pub const model = @import("model/model.zig");
pub const normalizer = @import("normalizer/normalizer.zig");
pub const pretokenizer = @import("pretokenizer/pretokenizer.zig");
pub const decoder = @import("decoder/decoder.zig");
pub const processor = @import("processor/processor.zig");
pub const config = @import("config.zig");
pub const types = @import("types.zig");

// Re-export common types from types.zig
pub const Offset = types.Offset;
pub const AddedToken = types.AddedToken;
pub const PaddingDirection = types.PaddingDirection;
pub const PaddingParams = types.PaddingParams;
pub const TruncationStrategy = types.TruncationStrategy;
pub const TruncationParams = types.TruncationParams;

/// The main Tokenizer struct
pub const Tokenizer = struct {
    allocator: std.mem.Allocator,
    model_impl: model.Model,
    model_ptr: ?*anyopaque = null, // Pointer to allocated model for cleanup
    added_vocab: Vocab,
    normalizer_impl: ?normalizer.Normalizer = null,
    pretokenizer_impl: ?pretokenizer.PreTokenizer = null,
    post_processor: ?processor.PostProcessor = null,
    decoder_impl: ?decoder.Decoder = null,
    truncation: ?TruncationParams = null,
    padding: ?PaddingParams = null,
    config_added_tokens: []AddedToken = &.{}, // Store config's added tokens for cleanup

    const Self = @This();

    /// Initialize from a tokenizer.json file
    pub fn fromFile(allocator: std.mem.Allocator, path: []const u8) !Self {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const content = try file.readToEndAlloc(allocator, 100 * 1024 * 1024);
        defer allocator.free(content);

        return fromJson(allocator, content);
    }

    /// Initialize from JSON content
    pub fn fromJson(allocator: std.mem.Allocator, json_content: []const u8) !Self {
        var cfg = try config.loadConfig(allocator, json_content);
        errdefer cfg.deinit();

        var added_vocab = Vocab.init(allocator);

        // Add tokens from config
        for (cfg.added_tokens) |token| {
            if (token.special) {
                _ = try added_vocab.addSpecialToken(token);
            } else {
                _ = try added_vocab.addToken(token);
            }
        }

        return .{
            .allocator = allocator,
            .model_impl = cfg.model_impl,
            .model_ptr = cfg.model_ptr,
            .added_vocab = added_vocab,
            .normalizer_impl = cfg.normalizer_impl,
            .pretokenizer_impl = cfg.pretokenizer_impl,
            .decoder_impl = cfg.decoder_impl,
            .post_processor = cfg.post_processor,
            .config_added_tokens = cfg.added_tokens, // Store for cleanup
        };
    }

    pub fn deinit(self: *Self) void {
        // Free config added tokens (content strings and array)
        for (self.config_added_tokens) |token| {
            self.allocator.free(token.content);
        }
        if (self.config_added_tokens.len > 0) {
            self.allocator.free(self.config_added_tokens);
        }

        // Free model via its deinit function (frees internal data structures)
        if (self.model_impl.deinit) |deinit_fn| {
            deinit_fn(self.model_impl.ptr);
        }
        // Destroy the model struct itself
        if (self.model_impl.destroy) |destroy_fn| {
            destroy_fn(self.allocator, self.model_impl.ptr);
        }

        self.added_vocab.deinit();
    }

    /// Encode text to tokens
    pub fn encode(self: *Self, text: []const u8, add_special_tokens: bool) !Encoding {
        var normalized: []const u8 = text;
        var normalized_allocated = false;

        // Step 1: Normalize
        if (self.normalizer_impl) |norm| {
            normalized = try norm.normalize(self.allocator, text);
            normalized_allocated = true;
        }
        defer if (normalized_allocated) self.allocator.free(normalized);

        // Step 2: Pre-tokenize
        var pretokens: []const []const u8 = &.{normalized};
        var pretokens_allocated = false;
        if (self.pretokenizer_impl) |pretok| {
            pretokens = try pretok.preTokenize(self.allocator, normalized);
            pretokens_allocated = true;
        }
        defer if (pretokens_allocated) self.allocator.free(pretokens);

        // Step 3: Tokenize each pre-token
        var all_tokens = std.ArrayListUnmanaged(Token){};
        defer all_tokens.deinit(self.allocator);

        for (pretokens) |pretoken| {
            const tokens = try self.model_impl.tokenize(self.allocator, pretoken);
            defer self.allocator.free(tokens);
            try all_tokens.appendSlice(self.allocator, tokens);
        }

        // Step 4: Create encoding
        var encoding = try Encoding.fromTokens(self.allocator, all_tokens.items);

        // Step 5: Post-process (add special tokens)
        if (add_special_tokens) {
            if (self.post_processor) |pp| {
                try pp.process(&encoding);
            }
        }

        // Step 6: Truncate
        if (self.truncation) |trunc| {
            try encoding.truncate(trunc.max_length, trunc.stride);
        }

        // Step 7: Pad
        if (self.padding) |pad| {
            try encoding.pad(self.allocator, pad);
        }

        return encoding;
    }

    /// Decode token IDs back to text
    pub fn decode(self: *Self, ids: []const u32, skip_special_tokens: bool) ![]u8 {
        var result = std.ArrayListUnmanaged(u8){};
        errdefer result.deinit(self.allocator);

        for (ids) |id| {
            if (skip_special_tokens) {
                if (self.added_vocab.idToToken(id)) |token| {
                    if (self.added_vocab.isSpecialToken(token)) {
                        continue;
                    }
                }
            }

            if (self.model_impl.idToToken(id)) |token| {
                try result.appendSlice(self.allocator, token);
            }
        }

        // Apply decoder if present
        if (self.decoder_impl) |dec| {
            const decoded = try dec.decode(self.allocator, result.items);
            result.deinit(self.allocator);
            return decoded;
        }

        return try result.toOwnedSlice(self.allocator);
    }

    /// Add special tokens to vocabulary
    pub fn addSpecialTokens(self: *Self, tokens: []const AddedToken) !usize {
        var added: usize = 0;
        for (tokens) |token| {
            if (try self.added_vocab.addSpecialToken(token)) {
                added += 1;
            }
        }
        return added;
    }

    /// Get vocabulary size
    pub fn getVocabSize(self: *const Self) usize {
        return self.model_impl.getVocabSize() + self.added_vocab.len();
    }

    /// Convert token to ID
    pub fn tokenToId(self: *const Self, token: []const u8) ?u32 {
        // Check added vocab first
        if (self.added_vocab.tokenToId(token)) |id| {
            return id;
        }
        return self.model_impl.tokenToId(token);
    }

    /// Convert ID to token
    pub fn idToToken(self: *const Self, id: u32) ?[]const u8 {
        // Check added vocab first
        if (self.added_vocab.idToToken(id)) |token| {
            return token;
        }
        return self.model_impl.idToToken(id);
    }
};

test "basic tokenizer" {
    // Basic compilation test
    const allocator = std.testing.allocator;
    _ = allocator;
}
