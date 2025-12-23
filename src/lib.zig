//! tokenizer-zig: Pure Zig implementation of HuggingFace tokenizers
//!
//! This library provides tokenization functionality compatible with
//! HuggingFace tokenizer.json files, supporting WordPiece (BERT) and
//! BPE (GPT-2, RoBERTa) tokenization models.

const std = @import("std");
pub const Token = @import("token.zig").Token;
pub const SpanToken = @import("token.zig").SpanToken;
pub const SpanTokenFlags = @import("token.zig").SpanTokenFlags;
pub const Encoding = @import("encoding.zig").Encoding;
pub const SpanEncoding = @import("encoding.zig").SpanEncoding;
pub const Vocab = @import("vocab.zig").Vocab;
pub const model = @import("model/model.zig");
pub const normalizer = @import("normalizer/normalizer.zig");
pub const pretokenizer = @import("pretokenizer/pretokenizer.zig");
pub const decoder = @import("decoder/decoder.zig");
pub const processor = @import("processor/processor.zig");
pub const config = @import("config.zig");
pub const types = @import("types.zig");
pub const arena = @import("arena.zig");

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

// ============================================================================
// FastTokenizer - Zero-Allocation Tokenizer
// ============================================================================

/// Model type for fast-path dispatch
pub const ModelType = enum {
    wordpiece,
    bpe,
};

/// Options for FastTokenizer initialization
pub const FastTokenizerOptions = struct {
    /// Maximum input sequence length in bytes
    max_sequence_length: u32 = 8192,
    /// Maximum output tokens
    max_tokens: u32 = 512,
};

/// Zero-allocation tokenizer using pre-allocated arenas.
/// After initialization, encode() performs zero allocations - it reuses
/// the pre-allocated arena buffers. The result is only valid until the
/// next encode() call.
pub const FastTokenizer = struct {
    /// Underlying tokenizer (for normalizer, pretokenizer, etc.)
    base: Tokenizer,
    /// Pre-allocated arena for zero-allocation encoding
    tokenizer_arena: *arena.TokenizerArena,
    /// Model type for fast-path dispatch
    model_type: ModelType,
    /// BPE model pointer (only valid if model_type == .bpe)
    bpe_model: ?*model.BPE = null,
    /// WordPiece model pointer (only valid if model_type == .wordpiece)
    wordpiece_model: ?*model.WordPiece = null,

    const Self = @This();

    /// Initialize from a tokenizer.json file with zero-allocation support.
    /// This is the only allocation point - after this, encode() is allocation-free.
    pub fn fromFile(allocator: std.mem.Allocator, path: []const u8, opts: FastTokenizerOptions) !Self {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const content = try file.readToEndAlloc(allocator, 100 * 1024 * 1024);
        defer allocator.free(content);

        return fromJson(allocator, content, opts);
    }

    /// Initialize from JSON content with zero-allocation support.
    pub fn fromJson(allocator: std.mem.Allocator, json_content: []const u8, opts: FastTokenizerOptions) !Self {
        // Load config to detect model type
        var cfg = try config.loadConfig(allocator, json_content);
        errdefer cfg.deinit();

        // Detect model type from config
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, json_content, .{}) catch {
            return error.InvalidJson;
        };
        defer parsed.deinit();

        const root = parsed.value;
        const model_obj = root.object.get("model") orelse return error.MissingModel;
        const model_type_str = if (model_obj.object.get("type")) |t| t.string else "WordPiece";

        const model_type: ModelType = if (std.mem.eql(u8, model_type_str, "BPE"))
            .bpe
        else
            .wordpiece;

        // Create base tokenizer from config
        var added_vocab = Vocab.init(allocator);
        for (cfg.added_tokens) |token| {
            if (token.special) {
                _ = try added_vocab.addSpecialToken(token);
            } else {
                _ = try added_vocab.addToken(token);
            }
        }

        const base = Tokenizer{
            .allocator = allocator,
            .model_impl = cfg.model_impl,
            .model_ptr = cfg.model_ptr,
            .added_vocab = added_vocab,
            .normalizer_impl = cfg.normalizer_impl,
            .pretokenizer_impl = cfg.pretokenizer_impl,
            .decoder_impl = cfg.decoder_impl,
            .post_processor = cfg.post_processor,
            .config_added_tokens = cfg.added_tokens,
        };

        // Create arena
        const tokenizer_arena = try allocator.create(arena.TokenizerArena);
        errdefer allocator.destroy(tokenizer_arena);

        tokenizer_arena.* = try arena.TokenizerArena.init(allocator, .{
            .max_sequence_length = opts.max_sequence_length,
            .max_tokens = opts.max_tokens,
        });

        // Get model pointers for fast path dispatch
        var bpe_model: ?*model.BPE = null;
        var wordpiece_model: ?*model.WordPiece = null;

        if (model_type == .bpe) {
            bpe_model = @ptrCast(@alignCast(cfg.model_ptr));
        } else if (model_type == .wordpiece) {
            wordpiece_model = @ptrCast(@alignCast(cfg.model_ptr));
        }

        return .{
            .base = base,
            .tokenizer_arena = tokenizer_arena,
            .model_type = model_type,
            .bpe_model = bpe_model,
            .wordpiece_model = wordpiece_model,
        };
    }

    /// Free all resources
    pub fn deinit(self: *Self) void {
        const allocator = self.base.allocator;
        self.tokenizer_arena.deinit();
        allocator.destroy(self.tokenizer_arena);
        self.base.deinit();
    }

    /// Zero-allocation encode - returns pointer to arena's encoding.
    /// IMPORTANT: The returned encoding is only valid until the next encode() call.
    /// If you need to keep the result, use encodeOwned() instead.
    pub fn encode(self: *Self, text: []const u8) !*SpanEncoding {
        var normalized: []const u8 = text;
        var normalized_allocated = false;
        const allocator = self.base.allocator;

        // Step 1: Normalize (may allocate, but typically doesn't for ASCII)
        if (self.base.normalizer_impl) |norm| {
            normalized = try norm.normalize(allocator, text);
            normalized_allocated = true;
        }
        defer if (normalized_allocated) allocator.free(normalized);

        // Reset arena for new encoding
        self.tokenizer_arena.reset(normalized);

        // Step 2: Pre-tokenize into arena scratch space
        if (self.base.pretokenizer_impl) |pretok| {
            const pretokens = try pretok.preTokenize(allocator, normalized);
            defer allocator.free(pretokens);

            for (pretokens) |pretoken| {
                // Calculate byte offset from pointer difference
                const start: u32 = @intCast(@intFromPtr(pretoken.ptr) - @intFromPtr(normalized.ptr));
                const end: u32 = @intCast(start + pretoken.len);
                self.tokenizer_arena.addPretokenSpan(start, end);
            }
        } else {
            // Whole input is one pre-token
            self.tokenizer_arena.addPretokenSpan(0, @intCast(normalized.len));
        }

        // Step 3: Tokenize each pre-token
        const pretoken_spans = self.tokenizer_arena.getPretokenSpans();

        if (self.model_type == .bpe and self.bpe_model != null) {
            // Fast path: O(n log n) BPE using arena
            for (pretoken_spans) |span| {
                const sequence = normalized[span[0]..span[1]];
                self.bpe_model.?.tokenizeFast(self.tokenizer_arena, sequence);
            }
        } else if (self.model_type == .wordpiece and self.wordpiece_model != null) {
            // Fast path: Zero-allocation WordPiece using arena
            for (pretoken_spans) |span| {
                const sequence = normalized[span[0]..span[1]];
                self.wordpiece_model.?.tokenizeFast(self.tokenizer_arena, sequence);
            }
        } else {
            // Fallback path: allocating tokenize (shouldn't reach here normally)
            for (pretoken_spans) |span| {
                const sequence = normalized[span[0]..span[1]];
                const tokens = try self.base.model_impl.tokenize(allocator, sequence);
                defer allocator.free(tokens);

                for (tokens) |token| {
                    if (!self.tokenizer_arena.encoding.tryAppend(SpanToken.init(
                        token.id,
                        token.offset.start + span[0],
                        token.offset.end + span[0],
                    ))) {
                        break; // Buffer full, stop adding tokens
                    }
                }
            }
        }

        return &self.tokenizer_arena.encoding;
    }

    /// Allocating encode - returns owned Encoding that caller must free.
    /// Use this when you need to keep the encoding beyond the next encode() call.
    pub fn encodeOwned(self: *Self, text: []const u8, add_special_tokens: bool) !Encoding {
        return self.base.encode(text, add_special_tokens);
    }

    /// Decode token IDs back to text (delegates to base tokenizer)
    pub fn decode(self: *Self, ids: []const u32, skip_special_tokens: bool) ![]u8 {
        return self.base.decode(ids, skip_special_tokens);
    }

    /// Get vocabulary size
    pub fn getVocabSize(self: *const Self) usize {
        return self.base.getVocabSize();
    }

    /// Convert token to ID
    pub fn tokenToId(self: *const Self, token: []const u8) ?u32 {
        return self.base.tokenToId(token);
    }

    /// Convert ID to token
    pub fn idToToken(self: *const Self, id: u32) ?[]const u8 {
        return self.base.idToToken(id);
    }

    /// Get memory usage of this tokenizer's arena
    pub fn arenaMemoryUsage(self: *const Self) usize {
        return self.tokenizer_arena.memoryUsage();
    }
};

// ============================================================================
// Test References - include all module tests
// ============================================================================

test {
    // Reference all tests from submodules
    _ = @import("types.zig");
    _ = @import("token.zig");
    _ = @import("vocab.zig");
    _ = @import("encoding.zig");
    _ = @import("arena.zig");
    _ = @import("model/wordpiece.zig");
    _ = @import("model/bpe.zig");
    _ = @import("config.zig");
}

test "basic tokenizer" {
    // Basic compilation test
    const allocator = std.testing.allocator;
    _ = allocator;
}

// ============================================================================
// Integration Tests - Full Pipeline
// ============================================================================

test "integration: WordPiece tokenization pipeline" {
    const allocator = std.testing.allocator;

    // Full BERT-like tokenizer config
    const json_content =
        \\{
        \\  "model": {
        \\    "type": "WordPiece",
        \\    "vocab": {
        \\      "[PAD]": 0,
        \\      "[UNK]": 1,
        \\      "[CLS]": 2,
        \\      "[SEP]": 3,
        \\      "hello": 4,
        \\      "world": 5,
        \\      "un": 6,
        \\      "##known": 7,
        \\      "play": 8,
        \\      "##ing": 9
        \\    },
        \\    "unk_token": "[UNK]",
        \\    "continuing_subword_prefix": "##"
        \\  },
        \\  "normalizer": {
        \\    "type": "BertNormalizer"
        \\  },
        \\  "pre_tokenizer": {
        \\    "type": "BertPreTokenizer"
        \\  },
        \\  "decoder": {
        \\    "type": "WordPiece"
        \\  },
        \\  "added_tokens": [
        \\    {"id": 2, "content": "[CLS]", "special": true},
        \\    {"id": 3, "content": "[SEP]", "special": true}
        \\  ]
        \\}
    ;

    var tokenizer = try Tokenizer.fromJson(allocator, json_content);
    defer tokenizer.deinit();

    // Test basic encoding
    var encoding = try tokenizer.encode("hello world", false);
    defer encoding.deinit();

    // Should have 2 tokens: "hello" and "world"
    try std.testing.expectEqual(@as(usize, 2), encoding.ids.len);
    try std.testing.expectEqual(@as(u32, 4), encoding.ids[0]); // "hello"
    try std.testing.expectEqual(@as(u32, 5), encoding.ids[1]); // "world"

    // Test vocab size includes added tokens
    try std.testing.expectEqual(@as(usize, 12), tokenizer.getVocabSize()); // 10 vocab + 2 added

    // Test tokenToId
    try std.testing.expectEqual(@as(?u32, 4), tokenizer.tokenToId("hello"));
    try std.testing.expectEqual(@as(?u32, 2), tokenizer.tokenToId("[CLS]"));

    // Test idToToken
    try std.testing.expectEqualStrings("hello", tokenizer.idToToken(4).?);
    try std.testing.expectEqualStrings("[CLS]", tokenizer.idToToken(2).?);
}

test "integration: BPE tokenization pipeline" {
    const allocator = std.testing.allocator;

    // GPT-2 style BPE tokenizer
    const json_content =
        \\{
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {
        \\      "<|endoftext|>": 0,
        \\      "h": 1,
        \\      "e": 2,
        \\      "l": 3,
        \\      "o": 4,
        \\      " ": 5,
        \\      "w": 6,
        \\      "r": 7,
        \\      "d": 8,
        \\      "he": 9,
        \\      "ll": 10,
        \\      "lo": 11
        \\    },
        \\    "merges": [
        \\      "h e",
        \\      "l l",
        \\      "l o"
        \\    ]
        \\  },
        \\  "pre_tokenizer": {
        \\    "type": "Whitespace"
        \\  },
        \\  "decoder": {
        \\    "type": "BPE"
        \\  }
        \\}
    ;

    var tokenizer = try Tokenizer.fromJson(allocator, json_content);
    defer tokenizer.deinit();

    // Test vocab size
    try std.testing.expectEqual(@as(usize, 12), tokenizer.getVocabSize());

    // Test tokenToId
    try std.testing.expectEqual(@as(?u32, 1), tokenizer.tokenToId("h"));
    try std.testing.expectEqual(@as(?u32, 9), tokenizer.tokenToId("he"));
}

test "integration: encoding with attention mask" {
    const allocator = std.testing.allocator;

    const json_content =
        \\{
        \\  "model": {
        \\    "type": "WordPiece",
        \\    "vocab": {
        \\      "[PAD]": 0,
        \\      "[UNK]": 1,
        \\      "test": 2,
        \\      "word": 3
        \\    }
        \\  }
        \\}
    ;

    var tokenizer = try Tokenizer.fromJson(allocator, json_content);
    defer tokenizer.deinit();

    var encoding = try tokenizer.encode("test", false);
    defer encoding.deinit();

    // Check attention mask (all 1s for actual tokens)
    try std.testing.expectEqual(@as(usize, 1), encoding.attention_mask.len);
    try std.testing.expectEqual(@as(u32, 1), encoding.attention_mask[0]);
}

test "integration: tokenizer decode" {
    const allocator = std.testing.allocator;

    const json_content =
        \\{
        \\  "model": {
        \\    "type": "WordPiece",
        \\    "vocab": {
        \\      "[PAD]": 0,
        \\      "[UNK]": 1,
        \\      "hello": 2,
        \\      "world": 3
        \\    }
        \\  },
        \\  "decoder": {
        \\    "type": "WordPiece"
        \\  }
        \\}
    ;

    var tokenizer = try Tokenizer.fromJson(allocator, json_content);
    defer tokenizer.deinit();

    // Decode token IDs
    const ids = [_]u32{ 2, 3 };
    const decoded = try tokenizer.decode(&ids, false);
    defer allocator.free(decoded);

    try std.testing.expectEqualStrings("helloworld", decoded);
}

test "integration: tokenizer skip special tokens in decode" {
    const allocator = std.testing.allocator;

    const json_content =
        \\{
        \\  "model": {
        \\    "type": "WordPiece",
        \\    "vocab": {
        \\      "[PAD]": 0,
        \\      "[CLS]": 1,
        \\      "[SEP]": 2,
        \\      "hello": 3
        \\    }
        \\  },
        \\  "added_tokens": [
        \\    {"id": 1, "content": "[CLS]", "special": true},
        \\    {"id": 2, "content": "[SEP]", "special": true}
        \\  ]
        \\}
    ;

    var tokenizer = try Tokenizer.fromJson(allocator, json_content);
    defer tokenizer.deinit();

    // Decode with special tokens included
    const ids_with_special = [_]u32{ 1, 3, 2 }; // [CLS] hello [SEP]
    const decoded_with_special = try tokenizer.decode(&ids_with_special, false);
    defer allocator.free(decoded_with_special);
    try std.testing.expectEqualStrings("[CLS]hello[SEP]", decoded_with_special);

    // Decode with special tokens skipped
    const decoded_skip_special = try tokenizer.decode(&ids_with_special, true);
    defer allocator.free(decoded_skip_special);
    try std.testing.expectEqualStrings("hello", decoded_skip_special);
}

test "integration: encoding offsets track original positions" {
    const allocator = std.testing.allocator;

    const json_content =
        \\{
        \\  "model": {
        \\    "type": "WordPiece",
        \\    "vocab": {
        \\      "[UNK]": 0,
        \\      "hello": 1
        \\    }
        \\  }
        \\}
    ;

    var tokenizer = try Tokenizer.fromJson(allocator, json_content);
    defer tokenizer.deinit();

    var encoding = try tokenizer.encode("hello", false);
    defer encoding.deinit();

    try std.testing.expectEqual(@as(usize, 1), encoding.offsets.len);
    try std.testing.expectEqual(@as(u32, 0), encoding.offsets[0].start);
    try std.testing.expectEqual(@as(u32, 5), encoding.offsets[0].end);
}

test "integration: add special tokens to vocabulary" {
    const allocator = std.testing.allocator;

    const json_content =
        \\{
        \\  "model": {
        \\    "type": "WordPiece",
        \\    "vocab": {
        \\      "[UNK]": 0,
        \\      "hello": 1
        \\    }
        \\  }
        \\}
    ;

    var tokenizer = try Tokenizer.fromJson(allocator, json_content);
    defer tokenizer.deinit();

    const initial_size = tokenizer.getVocabSize();

    // Add special tokens
    const new_tokens = [_]AddedToken{
        AddedToken.init("[MASK]", true),
        AddedToken.init("[NEW]", true),
    };
    const added = try tokenizer.addSpecialTokens(&new_tokens);

    try std.testing.expectEqual(@as(usize, 2), added);
    try std.testing.expectEqual(initial_size + 2, tokenizer.getVocabSize());

    // Verify they're accessible
    try std.testing.expect(tokenizer.tokenToId("[MASK]") != null);
    try std.testing.expect(tokenizer.tokenToId("[NEW]") != null);
}

test "integration: full BERT pipeline with normalizer and pretokenizer" {
    const allocator = std.testing.allocator;

    // Complete BERT-style config with all components
    const json_content =
        \\{
        \\  "model": {
        \\    "type": "WordPiece",
        \\    "vocab": {
        \\      "[PAD]": 0,
        \\      "[UNK]": 1,
        \\      "[CLS]": 2,
        \\      "[SEP]": 3,
        \\      "hello": 4,
        \\      "world": 5,
        \\      "test": 6,
        \\      ",": 7,
        \\      ".": 8,
        \\      "!": 9
        \\    },
        \\    "unk_token": "[UNK]",
        \\    "continuing_subword_prefix": "##"
        \\  },
        \\  "normalizer": {
        \\    "type": "BertNormalizer",
        \\    "lowercase": true,
        \\    "strip_accents": true,
        \\    "clean_text": true
        \\  },
        \\  "pre_tokenizer": {
        \\    "type": "BertPreTokenizer"
        \\  },
        \\  "decoder": {
        \\    "type": "WordPiece",
        \\    "prefix": "##"
        \\  },
        \\  "added_tokens": [
        \\    {"id": 2, "content": "[CLS]", "special": true},
        \\    {"id": 3, "content": "[SEP]", "special": true}
        \\  ]
        \\}
    ;

    var tokenizer = try Tokenizer.fromJson(allocator, json_content);
    defer tokenizer.deinit();

    // Test with mixed case and punctuation - normalizer should lowercase
    var encoding = try tokenizer.encode("Hello, World!", false);
    defer encoding.deinit();

    // Should tokenize: "hello", ",", "world", "!"
    try std.testing.expectEqual(@as(usize, 4), encoding.ids.len);
    try std.testing.expectEqual(@as(u32, 4), encoding.ids[0]); // "hello"
    try std.testing.expectEqual(@as(u32, 7), encoding.ids[1]); // ","
    try std.testing.expectEqual(@as(u32, 5), encoding.ids[2]); // "world"
    try std.testing.expectEqual(@as(u32, 9), encoding.ids[3]); // "!"
}

test "integration: encode empty string" {
    const allocator = std.testing.allocator;

    const json_content =
        \\{
        \\  "model": {
        \\    "type": "WordPiece",
        \\    "vocab": {
        \\      "[UNK]": 0,
        \\      "test": 1
        \\    }
        \\  }
        \\}
    ;

    var tokenizer = try Tokenizer.fromJson(allocator, json_content);
    defer tokenizer.deinit();

    var encoding = try tokenizer.encode("", false);
    defer encoding.deinit();

    // Empty input should produce empty encoding
    try std.testing.expectEqual(@as(usize, 0), encoding.ids.len);
    try std.testing.expectEqual(@as(usize, 0), encoding.tokens.len);
}

test "integration: decode empty ids" {
    const allocator = std.testing.allocator;

    const json_content =
        \\{
        \\  "model": {
        \\    "type": "WordPiece",
        \\    "vocab": {
        \\      "[UNK]": 0,
        \\      "test": 1
        \\    }
        \\  }
        \\}
    ;

    var tokenizer = try Tokenizer.fromJson(allocator, json_content);
    defer tokenizer.deinit();

    const empty_ids = [_]u32{};
    const decoded = try tokenizer.decode(&empty_ids, false);
    defer allocator.free(decoded);

    try std.testing.expectEqualStrings("", decoded);
}

test "integration: unknown tokens mapped to UNK" {
    const allocator = std.testing.allocator;

    const json_content =
        \\{
        \\  "model": {
        \\    "type": "WordPiece",
        \\    "vocab": {
        \\      "[UNK]": 0,
        \\      "hello": 1
        \\    },
        \\    "unk_token": "[UNK]"
        \\  }
        \\}
    ;

    var tokenizer = try Tokenizer.fromJson(allocator, json_content);
    defer tokenizer.deinit();

    // "goodbye" is not in vocab, should map to [UNK]
    var encoding = try tokenizer.encode("goodbye", false);
    defer encoding.deinit();

    try std.testing.expectEqual(@as(usize, 1), encoding.ids.len);
    try std.testing.expectEqual(@as(u32, 0), encoding.ids[0]); // [UNK]
}

test "integration: subword tokenization" {
    const allocator = std.testing.allocator;

    const json_content =
        \\{
        \\  "model": {
        \\    "type": "WordPiece",
        \\    "vocab": {
        \\      "[UNK]": 0,
        \\      "play": 1,
        \\      "##ing": 2,
        \\      "##ed": 3
        \\    },
        \\    "unk_token": "[UNK]",
        \\    "continuing_subword_prefix": "##"
        \\  }
        \\}
    ;

    var tokenizer = try Tokenizer.fromJson(allocator, json_content);
    defer tokenizer.deinit();

    // "playing" should be split into "play" + "##ing"
    var encoding = try tokenizer.encode("playing", false);
    defer encoding.deinit();

    try std.testing.expectEqual(@as(usize, 2), encoding.ids.len);
    try std.testing.expectEqual(@as(u32, 1), encoding.ids[0]); // "play"
    try std.testing.expectEqual(@as(u32, 2), encoding.ids[1]); // "##ing"
}

test "integration: multiple words with subword splits" {
    const allocator = std.testing.allocator;

    const json_content =
        \\{
        \\  "model": {
        \\    "type": "WordPiece",
        \\    "vocab": {
        \\      "[UNK]": 0,
        \\      "un": 1,
        \\      "##believ": 2,
        \\      "##able": 3,
        \\      "story": 4
        \\    },
        \\    "unk_token": "[UNK]",
        \\    "continuing_subword_prefix": "##"
        \\  },
        \\  "pre_tokenizer": {
        \\    "type": "Whitespace"
        \\  }
        \\}
    ;

    var tokenizer = try Tokenizer.fromJson(allocator, json_content);
    defer tokenizer.deinit();

    // "unbelievable story" => "un" + "##believ" + "##able" + "story"
    var encoding = try tokenizer.encode("unbelievable story", false);
    defer encoding.deinit();

    try std.testing.expectEqual(@as(usize, 4), encoding.ids.len);
    try std.testing.expectEqual(@as(u32, 1), encoding.ids[0]); // "un"
    try std.testing.expectEqual(@as(u32, 2), encoding.ids[1]); // "##believ"
    try std.testing.expectEqual(@as(u32, 3), encoding.ids[2]); // "##able"
    try std.testing.expectEqual(@as(u32, 4), encoding.ids[3]); // "story"
}

// ============================================================================
// FastTokenizer Tests
// ============================================================================

test "FastTokenizer: basic BPE encoding" {
    const allocator = std.testing.allocator;

    const json_content =
        \\{
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {
        \\      "h": 0,
        \\      "e": 1,
        \\      "l": 2,
        \\      "o": 3,
        \\      "he": 4,
        \\      "ll": 5,
        \\      "lo": 6
        \\    },
        \\    "merges": [
        \\      "h e",
        \\      "l l",
        \\      "l o"
        \\    ]
        \\  }
        \\}
    ;

    var tokenizer = try FastTokenizer.fromJson(allocator, json_content, .{});
    defer tokenizer.deinit();

    // Encode "hello"
    const encoding = try tokenizer.encode("hello");

    // Should have tokens: "he" + "ll" + "o"
    try std.testing.expect(encoding.len >= 1);
    try std.testing.expectEqual(ModelType.bpe, tokenizer.model_type);
}

test "FastTokenizer: WordPiece fallback" {
    const allocator = std.testing.allocator;

    const json_content =
        \\{
        \\  "model": {
        \\    "type": "WordPiece",
        \\    "vocab": {
        \\      "[UNK]": 0,
        \\      "hello": 1,
        \\      "world": 2
        \\    }
        \\  }
        \\}
    ;

    var tokenizer = try FastTokenizer.fromJson(allocator, json_content, .{});
    defer tokenizer.deinit();

    const encoding = try tokenizer.encode("hello");

    try std.testing.expectEqual(@as(u32, 1), encoding.len);
    try std.testing.expectEqual(@as(u32, 1), encoding.ids[0]); // "hello"
    try std.testing.expectEqual(ModelType.wordpiece, tokenizer.model_type);
}

test "FastTokenizer: arena memory usage" {
    const allocator = std.testing.allocator;

    const json_content =
        \\{
        \\  "model": {
        \\    "type": "WordPiece",
        \\    "vocab": {
        \\      "[UNK]": 0,
        \\      "test": 1
        \\    }
        \\  }
        \\}
    ;

    var tokenizer = try FastTokenizer.fromJson(allocator, json_content, .{
        .max_sequence_length = 1024,
        .max_tokens = 128,
    });
    defer tokenizer.deinit();

    // Memory should be allocated upfront
    const memory = tokenizer.arenaMemoryUsage();
    try std.testing.expect(memory > 0);
}

test "FastTokenizer: multiple encodes reuse arena" {
    const allocator = std.testing.allocator;

    const json_content =
        \\{
        \\  "model": {
        \\    "type": "WordPiece",
        \\    "vocab": {
        \\      "[UNK]": 0,
        \\      "hello": 1,
        \\      "world": 2,
        \\      "test": 3
        \\    }
        \\  }
        \\}
    ;

    var tokenizer = try FastTokenizer.fromJson(allocator, json_content, .{});
    defer tokenizer.deinit();

    // First encode
    const enc1 = try tokenizer.encode("hello");
    try std.testing.expectEqual(@as(u32, 1), enc1.len);
    try std.testing.expectEqual(@as(u32, 1), enc1.ids[0]);

    // Second encode - should reuse same arena, previous result is invalidated
    const enc2 = try tokenizer.encode("world");
    try std.testing.expectEqual(@as(u32, 1), enc2.len);
    try std.testing.expectEqual(@as(u32, 2), enc2.ids[0]);

    // Third encode
    const enc3 = try tokenizer.encode("test");
    try std.testing.expectEqual(@as(u32, 1), enc3.len);
    try std.testing.expectEqual(@as(u32, 3), enc3.ids[0]);
}

test "FastTokenizer: zero-copy token access" {
    const allocator = std.testing.allocator;

    const json_content =
        \\{
        \\  "model": {
        \\    "type": "WordPiece",
        \\    "vocab": {
        \\      "[UNK]": 0,
        \\      "hello": 1,
        \\      "world": 2
        \\    }
        \\  },
        \\  "pre_tokenizer": {
        \\    "type": "Whitespace"
        \\  }
        \\}
    ;

    var tokenizer = try FastTokenizer.fromJson(allocator, json_content, .{});
    defer tokenizer.deinit();

    const encoding = try tokenizer.encode("hello world");

    try std.testing.expectEqual(@as(u32, 2), encoding.len);

    // Verify we can get token strings via zero-copy access
    try std.testing.expectEqual(@as(u32, 1), encoding.ids[0]); // "hello"
    try std.testing.expectEqual(@as(u32, 2), encoding.ids[1]); // "world"
}

test "FastTokenizer: WordPiece subword tokenization" {
    const allocator = std.testing.allocator;

    const json_content =
        \\{
        \\  "model": {
        \\    "type": "WordPiece",
        \\    "vocab": {
        \\      "[UNK]": 0,
        \\      "play": 1,
        \\      "##ing": 2,
        \\      "##ed": 3,
        \\      "un": 4,
        \\      "##known": 5
        \\    },
        \\    "unk_token": "[UNK]",
        \\    "continuing_subword_prefix": "##"
        \\  }
        \\}
    ;

    var tokenizer = try FastTokenizer.fromJson(allocator, json_content, .{});
    defer tokenizer.deinit();

    // Test subword tokenization: "playing" -> "play" + "##ing"
    const encoding = try tokenizer.encode("playing");

    try std.testing.expectEqual(@as(u32, 2), encoding.len);
    try std.testing.expectEqual(@as(u32, 1), encoding.ids[0]); // "play"
    try std.testing.expectEqual(@as(u32, 2), encoding.ids[1]); // "##ing"

    // Verify offsets
    try std.testing.expectEqual(@as(u32, 0), encoding.tokens[0].start);
    try std.testing.expectEqual(@as(u32, 4), encoding.tokens[0].end);
    try std.testing.expectEqual(@as(u32, 4), encoding.tokens[1].start);
    try std.testing.expectEqual(@as(u32, 7), encoding.tokens[1].end);
}

test "FastTokenizer: WordPiece unknown word returns UNK" {
    const allocator = std.testing.allocator;

    const json_content =
        \\{
        \\  "model": {
        \\    "type": "WordPiece",
        \\    "vocab": {
        \\      "[UNK]": 0,
        \\      "hello": 1
        \\    },
        \\    "unk_token": "[UNK]"
        \\  }
        \\}
    ;

    var tokenizer = try FastTokenizer.fromJson(allocator, json_content, .{});
    defer tokenizer.deinit();

    // "xyz" is not in vocab, should return UNK
    const encoding = try tokenizer.encode("xyz");

    try std.testing.expectEqual(@as(u32, 1), encoding.len);
    try std.testing.expectEqual(@as(u32, 0), encoding.ids[0]); // [UNK]
}
