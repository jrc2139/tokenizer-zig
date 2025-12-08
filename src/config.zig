//! JSON configuration loading for HuggingFace tokenizer.json files
//!
//! Parses the tokenizer.json format and constructs the appropriate
//! tokenizer components (model, normalizer, pre-tokenizer, post-processor, decoder).

const std = @import("std");
const model = @import("model/model.zig");
const wordpiece = @import("model/wordpiece.zig");
const bpe = @import("model/bpe.zig");
const normalizer = @import("normalizer/normalizer.zig");
const pretokenizer = @import("pretokenizer/pretokenizer.zig");
const decoder_mod = @import("decoder/decoder.zig");
const processor = @import("processor/processor.zig");
const Vocab = @import("vocab.zig").Vocab;
const types = @import("types.zig");
pub const AddedToken = types.AddedToken;

pub const ConfigError = error{
    InvalidJson,
    MissingModel,
    UnsupportedModelType,
    MissingVocab,
    InvalidVocabEntry,
    InvalidMergeEntry,
    OutOfMemory,
    UnsupportedNormalizer,
    UnsupportedPreTokenizer,
    UnsupportedDecoder,
    UnsupportedPostProcessor,
};

/// Parsed tokenizer configuration
pub const TokenizerConfig = struct {
    allocator: std.mem.Allocator,
    model_impl: model.Model,
    model_ptr: *anyopaque, // Keep track of allocated model for cleanup
    normalizer_impl: ?normalizer.Normalizer = null,
    pretokenizer_impl: ?pretokenizer.PreTokenizer = null,
    decoder_impl: ?decoder_mod.Decoder = null,
    post_processor: ?processor.PostProcessor = null,
    added_tokens: []AddedToken = &.{},

    const Self = @This();

    pub fn deinit(self: *Self) void {
        // Free added tokens
        for (self.added_tokens) |token| {
            self.allocator.free(token.content);
        }
        if (self.added_tokens.len > 0) {
            self.allocator.free(self.added_tokens);
        }

        // Model cleanup is handled by the Tokenizer
    }
};

/// Load and parse tokenizer.json content
pub fn loadConfig(allocator: std.mem.Allocator, json_content: []const u8) !TokenizerConfig {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, json_content, .{}) catch {
        return ConfigError.InvalidJson;
    };
    defer parsed.deinit();

    const root = parsed.value;
    if (root != .object) {
        return ConfigError.InvalidJson;
    }

    var config = TokenizerConfig{
        .allocator = allocator,
        .model_impl = undefined,
        .model_ptr = undefined,
    };

    // Parse model
    const model_result = try parseModel(allocator, root.object);
    config.model_impl = model_result.model;
    config.model_ptr = model_result.ptr;

    // Parse added tokens
    if (root.object.get("added_tokens")) |added_tokens_val| {
        if (added_tokens_val == .array) {
            config.added_tokens = try parseAddedTokens(allocator, added_tokens_val.array);
        }
    }

    // Parse normalizer (optional)
    if (root.object.get("normalizer")) |norm_val| {
        if (norm_val != .null) {
            config.normalizer_impl = try parseNormalizer(allocator, norm_val);
        }
    }

    // Parse pre_tokenizer (optional)
    if (root.object.get("pre_tokenizer")) |pretok_val| {
        if (pretok_val != .null) {
            config.pretokenizer_impl = try parsePreTokenizer(allocator, pretok_val);
        }
    }

    // Parse decoder (optional)
    if (root.object.get("decoder")) |dec_val| {
        if (dec_val != .null) {
            config.decoder_impl = try parseDecoder(allocator, dec_val);
        }
    }

    // Parse post_processor (optional)
    if (root.object.get("post_processor")) |pp_val| {
        if (pp_val != .null) {
            config.post_processor = try parsePostProcessor(allocator, pp_val);
        }
    }

    return config;
}

const ModelResult = struct {
    model: model.Model,
    ptr: *anyopaque,
};

fn parseModel(allocator: std.mem.Allocator, root: std.json.ObjectMap) !ModelResult {
    const model_obj = root.get("model") orelse return ConfigError.MissingModel;
    if (model_obj != .object) {
        return ConfigError.MissingModel;
    }

    const model_type = getStringField(model_obj.object, "type") orelse "WordPiece";

    if (std.mem.eql(u8, model_type, "WordPiece")) {
        return try parseWordPieceModel(allocator, model_obj.object);
    } else if (std.mem.eql(u8, model_type, "BPE")) {
        return try parseBPEModel(allocator, model_obj.object);
    } else {
        return ConfigError.UnsupportedModelType;
    }
}

fn parseWordPieceModel(allocator: std.mem.Allocator, model_obj: std.json.ObjectMap) !ModelResult {
    // Parse vocab
    const vocab_val = model_obj.get("vocab") orelse return ConfigError.MissingVocab;
    if (vocab_val != .object) {
        return ConfigError.MissingVocab;
    }

    var vocab = std.StringHashMapUnmanaged(u32){};
    errdefer {
        var it = vocab.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
        }
        vocab.deinit(allocator);
    }

    var vocab_it = vocab_val.object.iterator();
    while (vocab_it.next()) |entry| {
        const key = entry.key_ptr.*;
        const val = entry.value_ptr.*;

        if (val != .integer) {
            return ConfigError.InvalidVocabEntry;
        }

        const id: u32 = @intCast(val.integer);
        const key_copy = try allocator.dupe(u8, key);
        try vocab.put(allocator, key_copy, id);
    }

    // Parse config options - must duplicate strings since JSON is freed after parsing
    const unk_token_raw = getStringField(model_obj, "unk_token") orelse "[UNK]";
    const prefix_raw = getStringField(model_obj, "continuing_subword_prefix") orelse "##";
    const max_chars: usize = if (model_obj.get("max_input_chars_per_word")) |v|
        if (v == .integer) @intCast(v.integer) else 100
    else
        100;

    // Duplicate strings so they outlive the JSON parse
    const unk_token = try allocator.dupe(u8, unk_token_raw);
    errdefer allocator.free(unk_token);
    const prefix = try allocator.dupe(u8, prefix_raw);
    errdefer allocator.free(prefix);

    const wp_ptr = try allocator.create(wordpiece.WordPiece);
    wp_ptr.* = try wordpiece.WordPiece.initOwned(allocator, vocab, unk_token, prefix, max_chars);

    return .{
        .model = wp_ptr.getModel(),
        .ptr = wp_ptr,
    };
}

fn parseBPEModel(allocator: std.mem.Allocator, model_obj: std.json.ObjectMap) !ModelResult {
    // Parse vocab
    const vocab_val = model_obj.get("vocab") orelse return ConfigError.MissingVocab;
    if (vocab_val != .object) {
        return ConfigError.MissingVocab;
    }

    var vocab = std.StringHashMapUnmanaged(u32){};
    errdefer {
        var it = vocab.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
        }
        vocab.deinit(allocator);
    }

    var vocab_it = vocab_val.object.iterator();
    while (vocab_it.next()) |entry| {
        const key = entry.key_ptr.*;
        const val = entry.value_ptr.*;

        if (val != .integer) {
            return ConfigError.InvalidVocabEntry;
        }

        const id: u32 = @intCast(val.integer);
        const key_copy = try allocator.dupe(u8, key);
        try vocab.put(allocator, key_copy, id);
    }

    // Parse merges
    var merges = std.AutoHashMapUnmanaged(u64, bpe.PairVal){};
    errdefer merges.deinit(allocator);

    if (model_obj.get("merges")) |merges_val| {
        if (merges_val == .array) {
            var rank: u32 = 0;
            for (merges_val.array.items) |merge_item| {
                // Merges can be either strings "token1 token2" or arrays ["token1", "token2"]
                var first: []const u8 = undefined;
                var second: []const u8 = undefined;

                if (merge_item == .string) {
                    // Parse "token1 token2" format
                    const merge_str = merge_item.string;
                    var parts = std.mem.splitScalar(u8, merge_str, ' ');
                    first = parts.next() orelse continue;
                    second = parts.next() orelse continue;
                } else if (merge_item == .array and merge_item.array.items.len == 2) {
                    // Parse ["token1", "token2"] format
                    if (merge_item.array.items[0] != .string or merge_item.array.items[1] != .string) {
                        continue;
                    }
                    first = merge_item.array.items[0].string;
                    second = merge_item.array.items[1].string;
                } else {
                    continue;
                }

                // Look up token IDs
                const first_id = vocab.get(first) orelse continue;
                const second_id = vocab.get(second) orelse continue;

                // Compute merged token and look up its ID
                var merged_buf: [512]u8 = undefined;
                const merged_len = first.len + second.len;
                if (merged_len > merged_buf.len) continue;

                @memcpy(merged_buf[0..first.len], first);
                @memcpy(merged_buf[first.len..merged_len], second);
                const merged = merged_buf[0..merged_len];

                const new_id = vocab.get(merged) orelse continue;

                const pair = bpe.Pair{ .first = first_id, .second = second_id };
                try merges.put(allocator, pair.hash(), .{ .rank = rank, .new_id = new_id });
                rank += 1;
            }
        }
    }

    // Parse config options - must duplicate strings since JSON is freed after parsing
    const unk_token_raw = getStringField(model_obj, "unk_token");
    const prefix_raw = getStringField(model_obj, "continuing_subword_prefix");
    const suffix_raw = getStringField(model_obj, "end_of_word_suffix");

    // Duplicate strings so they outlive the JSON parse
    const unk_token: ?[]const u8 = if (unk_token_raw) |s| try allocator.dupe(u8, s) else null;
    errdefer if (unk_token) |s| allocator.free(s);
    const prefix: ?[]const u8 = if (prefix_raw) |s| try allocator.dupe(u8, s) else null;
    errdefer if (prefix) |s| allocator.free(s);
    const suffix: ?[]const u8 = if (suffix_raw) |s| try allocator.dupe(u8, s) else null;
    errdefer if (suffix) |s| allocator.free(s);

    const bpe_ptr = try allocator.create(bpe.BPE);
    bpe_ptr.* = try bpe.BPE.initOwned(allocator, vocab, merges, unk_token, prefix, suffix);

    return .{
        .model = bpe_ptr.getModel(),
        .ptr = bpe_ptr,
    };
}

fn parseAddedTokens(allocator: std.mem.Allocator, tokens_array: std.json.Array) ![]AddedToken {
    var tokens = try allocator.alloc(AddedToken, tokens_array.items.len);
    errdefer allocator.free(tokens);

    var count: usize = 0;
    for (tokens_array.items) |item| {
        if (item != .object) continue;

        const obj = item.object;
        const content = getStringField(obj, "content") orelse continue;

        const id: ?u32 = if (obj.get("id")) |id_val|
            if (id_val == .integer) @intCast(id_val.integer) else null
        else
            null;

        const special = getBoolField(obj, "special") orelse false;
        const single_word = getBoolField(obj, "single_word") orelse false;
        const lstrip = getBoolField(obj, "lstrip") orelse false;
        const rstrip = getBoolField(obj, "rstrip") orelse false;
        const normalized = getBoolField(obj, "normalized") orelse true;

        tokens[count] = .{
            .content = try allocator.dupe(u8, content),
            .id = id,
            .special = special,
            .single_word = single_word,
            .lstrip = lstrip,
            .rstrip = rstrip,
            .normalized = normalized,
        };
        count += 1;
    }

    // Resize to actual count
    if (count < tokens.len) {
        tokens = try allocator.realloc(tokens, count);
    }

    return tokens;
}

fn parseNormalizer(allocator: std.mem.Allocator, val: std.json.Value) !?normalizer.Normalizer {
    _ = allocator;
    if (val != .object) return null;

    const norm_type = getStringField(val.object, "type") orelse return null;

    if (std.mem.eql(u8, norm_type, "BertNormalizer")) {
        // BertNormalizer is the default, just return a basic normalizer
        return normalizer.Normalizer{
            .ptr = undefined,
            .normalize_fn = bertNormalizeImpl,
            .deinit_fn = null,
        };
    } else if (std.mem.eql(u8, norm_type, "Lowercase")) {
        return normalizer.Normalizer{
            .ptr = undefined,
            .normalize_fn = lowercaseNormalizeImpl,
            .deinit_fn = null,
        };
    }

    // For unsupported normalizers, return null (skip normalization)
    return null;
}

fn bertNormalizeImpl(_: *anyopaque, allocator: std.mem.Allocator, input: []const u8) anyerror![]const u8 {
    // Simple BERT normalization: lowercase and basic cleanup
    var result = try allocator.alloc(u8, input.len);
    for (input, 0..) |c, i| {
        result[i] = std.ascii.toLower(c);
    }
    return result;
}

fn lowercaseNormalizeImpl(_: *anyopaque, allocator: std.mem.Allocator, input: []const u8) anyerror![]const u8 {
    var result = try allocator.alloc(u8, input.len);
    for (input, 0..) |c, i| {
        result[i] = std.ascii.toLower(c);
    }
    return result;
}

fn parsePreTokenizer(allocator: std.mem.Allocator, val: std.json.Value) !?pretokenizer.PreTokenizer {
    _ = allocator;
    if (val != .object) return null;

    const pretok_type = getStringField(val.object, "type") orelse return null;

    if (std.mem.eql(u8, pretok_type, "BertPreTokenizer")) {
        return pretokenizer.PreTokenizer{
            .ptr = undefined,
            .pre_tokenize_fn = bertPreTokenizeImpl,
            .deinit_fn = null,
        };
    } else if (std.mem.eql(u8, pretok_type, "Whitespace") or std.mem.eql(u8, pretok_type, "WhitespaceSplit")) {
        return pretokenizer.PreTokenizer{
            .ptr = undefined,
            .pre_tokenize_fn = whitespacePreTokenizeImpl,
            .deinit_fn = null,
        };
    }

    // For unsupported pre-tokenizers, return null
    return null;
}

fn bertPreTokenizeImpl(_: *anyopaque, allocator: std.mem.Allocator, input: []const u8) anyerror![]const []const u8 {
    // Split on whitespace and punctuation
    var tokens = std.ArrayListUnmanaged([]const u8){};
    errdefer tokens.deinit(allocator);

    var start: usize = 0;
    var i: usize = 0;

    while (i < input.len) {
        const c = input[i];
        const is_whitespace = std.ascii.isWhitespace(c);
        const is_punct = isPunctuation(c);

        if (is_whitespace or is_punct) {
            // Add token before this character
            if (i > start) {
                try tokens.append(allocator, input[start..i]);
            }
            // Add punctuation as separate token
            if (is_punct) {
                try tokens.append(allocator, input[i .. i + 1]);
            }
            start = i + 1;
        }
        i += 1;
    }

    // Add final token
    if (start < input.len) {
        try tokens.append(allocator, input[start..]);
    }

    return try tokens.toOwnedSlice(allocator);
}

fn whitespacePreTokenizeImpl(_: *anyopaque, allocator: std.mem.Allocator, input: []const u8) anyerror![]const []const u8 {
    var tokens = std.ArrayListUnmanaged([]const u8){};
    errdefer tokens.deinit(allocator);

    var iter = std.mem.tokenizeAny(u8, input, " \t\n\r");
    while (iter.next()) |token| {
        try tokens.append(allocator, token);
    }

    return try tokens.toOwnedSlice(allocator);
}

fn isPunctuation(c: u8) bool {
    return switch (c) {
        '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~' => true,
        else => false,
    };
}

fn parseDecoder(allocator: std.mem.Allocator, val: std.json.Value) !?decoder_mod.Decoder {
    _ = allocator;
    if (val != .object) return null;

    const dec_type = getStringField(val.object, "type") orelse return null;

    if (std.mem.eql(u8, dec_type, "WordPiece")) {
        return decoder_mod.Decoder{
            .ptr = undefined,
            .decode_fn = wordPieceDecodeImpl,
            .deinit_fn = null,
        };
    } else if (std.mem.eql(u8, dec_type, "ByteLevel")) {
        return decoder_mod.Decoder{
            .ptr = undefined,
            .decode_fn = byteLevelDecodeImpl,
            .deinit_fn = null,
        };
    } else if (std.mem.eql(u8, dec_type, "BPE")) {
        return decoder_mod.Decoder{
            .ptr = undefined,
            .decode_fn = bpeDecodeImpl,
            .deinit_fn = null,
        };
    }

    return null;
}

fn wordPieceDecodeImpl(_: *anyopaque, allocator: std.mem.Allocator, tokens: []const u8) anyerror![]u8 {
    // Remove "##" prefixes and join
    var result = std.ArrayListUnmanaged(u8){};
    errdefer result.deinit(allocator);

    var i: usize = 0;
    while (i < tokens.len) {
        if (i + 1 < tokens.len and tokens[i] == '#' and tokens[i + 1] == '#') {
            // Skip "##" prefix
            i += 2;
        } else {
            try result.append(allocator, tokens[i]);
            i += 1;
        }
    }

    return try result.toOwnedSlice(allocator);
}

fn byteLevelDecodeImpl(_: *anyopaque, allocator: std.mem.Allocator, tokens: []const u8) anyerror![]u8 {
    // For ByteLevel, just copy the input
    return try allocator.dupe(u8, tokens);
}

fn bpeDecodeImpl(_: *anyopaque, allocator: std.mem.Allocator, tokens: []const u8) anyerror![]u8 {
    // For BPE, replace special tokens like "Ġ" with space
    var result = std.ArrayListUnmanaged(u8){};
    errdefer result.deinit(allocator);

    var i: usize = 0;
    while (i < tokens.len) {
        // Check for UTF-8 "Ġ" (0xC4 0xA0)
        if (i + 1 < tokens.len and tokens[i] == 0xC4 and tokens[i + 1] == 0xA0) {
            try result.append(allocator, ' ');
            i += 2;
        } else {
            try result.append(allocator, tokens[i]);
            i += 1;
        }
    }

    return try result.toOwnedSlice(allocator);
}

fn parsePostProcessor(allocator: std.mem.Allocator, val: std.json.Value) !?processor.PostProcessor {
    _ = allocator;
    if (val != .object) return null;

    const pp_type = getStringField(val.object, "type") orelse return null;

    if (std.mem.eql(u8, pp_type, "TemplateProcessing") or std.mem.eql(u8, pp_type, "BertProcessing")) {
        // Parse special tokens from the config
        // For now, return a basic BERT-style processor
        return processor.PostProcessor{
            .ptr = undefined,
            .process_fn = bertPostProcessImpl,
            .deinit_fn = null,
        };
    }

    return null;
}

fn bertPostProcessImpl(_: *anyopaque, _: *@import("encoding.zig").Encoding) anyerror!void {
    // BERT post-processing adds [CLS] at start and [SEP] at end
    // This is handled in the Tokenizer.encode method if add_special_tokens is true
    // For now, this is a no-op as we handle it separately
}

// Helper functions
fn getStringField(obj: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    if (obj.get(key)) |val| {
        if (val == .string) {
            return val.string;
        }
    }
    return null;
}

fn getBoolField(obj: std.json.ObjectMap, key: []const u8) ?bool {
    if (obj.get(key)) |val| {
        if (val == .bool) {
            return val.bool;
        }
    }
    return null;
}

// ============================================================================
// Unit Tests
// ============================================================================

/// Helper to cleanup config after tests
fn cleanupConfig(config: *TokenizerConfig) void {
    config.deinit();
    // Clean up model - check type by testing if it's WordPiece or BPE
    if (config.model_impl.deinit) |deinit_fn| {
        deinit_fn(config.model_impl.ptr);
    }
    if (config.model_impl.destroy) |destroy_fn| {
        destroy_fn(config.allocator, config.model_impl.ptr);
    }
}

test "parse simple vocab" {
    const allocator = std.testing.allocator;

    const json_content =
        \\{
        \\  "version": "1.0",
        \\  "model": {
        \\    "type": "WordPiece",
        \\    "vocab": {
        \\      "[PAD]": 0,
        \\      "[UNK]": 1,
        \\      "hello": 2,
        \\      "world": 3
        \\    },
        \\    "unk_token": "[UNK]",
        \\    "continuing_subword_prefix": "##"
        \\  }
        \\}
    ;

    var config = try loadConfig(allocator, json_content);
    defer cleanupConfig(&config);

    // Verify vocab loaded
    try std.testing.expectEqual(@as(usize, 4), config.model_impl.getVocabSize());
    try std.testing.expectEqual(@as(?u32, 0), config.model_impl.tokenToId("[PAD]"));
    try std.testing.expectEqual(@as(?u32, 2), config.model_impl.tokenToId("hello"));
}

test "config invalid json returns error" {
    const allocator = std.testing.allocator;
    const result = loadConfig(allocator, "not valid json {{{");
    try std.testing.expectError(ConfigError.InvalidJson, result);
}

test "config missing model returns error" {
    const allocator = std.testing.allocator;
    const json_content =
        \\{
        \\  "version": "1.0"
        \\}
    ;
    const result = loadConfig(allocator, json_content);
    try std.testing.expectError(ConfigError.MissingModel, result);
}

test "config unsupported model type returns error" {
    const allocator = std.testing.allocator;
    const json_content =
        \\{
        \\  "model": {
        \\    "type": "UnknownModel",
        \\    "vocab": {}
        \\  }
        \\}
    ;
    const result = loadConfig(allocator, json_content);
    try std.testing.expectError(ConfigError.UnsupportedModelType, result);
}

test "config missing vocab returns error" {
    const allocator = std.testing.allocator;
    const json_content =
        \\{
        \\  "model": {
        \\    "type": "WordPiece"
        \\  }
        \\}
    ;
    const result = loadConfig(allocator, json_content);
    try std.testing.expectError(ConfigError.MissingVocab, result);
}

test "parse BPE model" {
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
        \\      "hello": 6
        \\    },
        \\    "merges": [
        \\      "h e",
        \\      "l l"
        \\    ]
        \\  }
        \\}
    ;

    var config = try loadConfig(allocator, json_content);
    defer cleanupConfig(&config);

    // Verify vocab loaded
    try std.testing.expectEqual(@as(usize, 7), config.model_impl.getVocabSize());
    try std.testing.expectEqual(@as(?u32, 0), config.model_impl.tokenToId("h"));
    try std.testing.expectEqual(@as(?u32, 4), config.model_impl.tokenToId("he"));
}

test "parse added tokens" {
    const allocator = std.testing.allocator;

    const json_content =
        \\{
        \\  "model": {
        \\    "type": "WordPiece",
        \\    "vocab": {
        \\      "[PAD]": 0,
        \\      "[UNK]": 1
        \\    }
        \\  },
        \\  "added_tokens": [
        \\    {
        \\      "id": 100,
        \\      "content": "[CLS]",
        \\      "special": true,
        \\      "single_word": false,
        \\      "lstrip": false,
        \\      "rstrip": false
        \\    },
        \\    {
        \\      "id": 101,
        \\      "content": "[SEP]",
        \\      "special": true
        \\    }
        \\  ]
        \\}
    ;

    var config = try loadConfig(allocator, json_content);
    defer cleanupConfig(&config);

    try std.testing.expectEqual(@as(usize, 2), config.added_tokens.len);
    try std.testing.expectEqualStrings("[CLS]", config.added_tokens[0].content);
    try std.testing.expectEqual(@as(?u32, 100), config.added_tokens[0].id);
    try std.testing.expect(config.added_tokens[0].special);
    try std.testing.expectEqualStrings("[SEP]", config.added_tokens[1].content);
    try std.testing.expectEqual(@as(?u32, 101), config.added_tokens[1].id);
}

test "parse normalizer BertNormalizer" {
    const allocator = std.testing.allocator;

    const json_content =
        \\{
        \\  "model": {
        \\    "type": "WordPiece",
        \\    "vocab": {"[UNK]": 0}
        \\  },
        \\  "normalizer": {
        \\    "type": "BertNormalizer",
        \\    "clean_text": true,
        \\    "handle_chinese_chars": true,
        \\    "strip_accents": true,
        \\    "lowercase": true
        \\  }
        \\}
    ;

    var config = try loadConfig(allocator, json_content);
    defer cleanupConfig(&config);

    try std.testing.expect(config.normalizer_impl != null);

    // Test the normalizer
    const normalized = try config.normalizer_impl.?.normalize(allocator, "HELLO");
    defer allocator.free(normalized);
    try std.testing.expectEqualStrings("hello", normalized);
}

test "parse normalizer Lowercase" {
    const allocator = std.testing.allocator;

    const json_content =
        \\{
        \\  "model": {
        \\    "type": "WordPiece",
        \\    "vocab": {"[UNK]": 0}
        \\  },
        \\  "normalizer": {
        \\    "type": "Lowercase"
        \\  }
        \\}
    ;

    var config = try loadConfig(allocator, json_content);
    defer cleanupConfig(&config);

    try std.testing.expect(config.normalizer_impl != null);

    const normalized = try config.normalizer_impl.?.normalize(allocator, "WORLD");
    defer allocator.free(normalized);
    try std.testing.expectEqualStrings("world", normalized);
}

test "parse pre_tokenizer BertPreTokenizer" {
    const allocator = std.testing.allocator;

    const json_content =
        \\{
        \\  "model": {
        \\    "type": "WordPiece",
        \\    "vocab": {"[UNK]": 0}
        \\  },
        \\  "pre_tokenizer": {
        \\    "type": "BertPreTokenizer"
        \\  }
        \\}
    ;

    var config = try loadConfig(allocator, json_content);
    defer cleanupConfig(&config);

    try std.testing.expect(config.pretokenizer_impl != null);

    // Test the pre-tokenizer
    const tokens = try config.pretokenizer_impl.?.preTokenize(allocator, "hello world");
    defer allocator.free(tokens);

    try std.testing.expectEqual(@as(usize, 2), tokens.len);
    try std.testing.expectEqualStrings("hello", tokens[0]);
    try std.testing.expectEqualStrings("world", tokens[1]);
}

test "parse pre_tokenizer Whitespace" {
    const allocator = std.testing.allocator;

    const json_content =
        \\{
        \\  "model": {
        \\    "type": "WordPiece",
        \\    "vocab": {"[UNK]": 0}
        \\  },
        \\  "pre_tokenizer": {
        \\    "type": "Whitespace"
        \\  }
        \\}
    ;

    var config = try loadConfig(allocator, json_content);
    defer cleanupConfig(&config);

    try std.testing.expect(config.pretokenizer_impl != null);

    const tokens = try config.pretokenizer_impl.?.preTokenize(allocator, "hello  world\ttest");
    defer allocator.free(tokens);

    try std.testing.expectEqual(@as(usize, 3), tokens.len);
    try std.testing.expectEqualStrings("hello", tokens[0]);
    try std.testing.expectEqualStrings("world", tokens[1]);
    try std.testing.expectEqualStrings("test", tokens[2]);
}

test "parse decoder WordPiece" {
    const allocator = std.testing.allocator;

    const json_content =
        \\{
        \\  "model": {
        \\    "type": "WordPiece",
        \\    "vocab": {"[UNK]": 0}
        \\  },
        \\  "decoder": {
        \\    "type": "WordPiece",
        \\    "prefix": "##"
        \\  }
        \\}
    ;

    var config = try loadConfig(allocator, json_content);
    defer cleanupConfig(&config);

    try std.testing.expect(config.decoder_impl != null);

    // Test the decoder - removes ## prefixes
    const decoded = try config.decoder_impl.?.decode(allocator, "play##ing");
    defer allocator.free(decoded);
    try std.testing.expectEqualStrings("playing", decoded);
}

test "parse decoder BPE" {
    const allocator = std.testing.allocator;

    const json_content =
        \\{
        \\  "model": {
        \\    "type": "WordPiece",
        \\    "vocab": {"[UNK]": 0}
        \\  },
        \\  "decoder": {
        \\    "type": "BPE"
        \\  }
        \\}
    ;

    var config = try loadConfig(allocator, json_content);
    defer cleanupConfig(&config);

    try std.testing.expect(config.decoder_impl != null);

    // Test the decoder - replaces Ġ with space
    const decoded = try config.decoder_impl.?.decode(allocator, "hello\xC4\xA0world");
    defer allocator.free(decoded);
    try std.testing.expectEqualStrings("hello world", decoded);
}

test "parse post_processor BertProcessing" {
    const allocator = std.testing.allocator;

    const json_content =
        \\{
        \\  "model": {
        \\    "type": "WordPiece",
        \\    "vocab": {"[UNK]": 0}
        \\  },
        \\  "post_processor": {
        \\    "type": "BertProcessing",
        \\    "sep": ["[SEP]", 102],
        \\    "cls": ["[CLS]", 101]
        \\  }
        \\}
    ;

    var config = try loadConfig(allocator, json_content);
    defer cleanupConfig(&config);

    try std.testing.expect(config.post_processor != null);
}

test "parse null normalizer" {
    const allocator = std.testing.allocator;

    const json_content =
        \\{
        \\  "model": {
        \\    "type": "WordPiece",
        \\    "vocab": {"[UNK]": 0}
        \\  },
        \\  "normalizer": null
        \\}
    ;

    var config = try loadConfig(allocator, json_content);
    defer cleanupConfig(&config);

    try std.testing.expect(config.normalizer_impl == null);
}

test "parse unsupported normalizer returns null" {
    const allocator = std.testing.allocator;

    const json_content =
        \\{
        \\  "model": {
        \\    "type": "WordPiece",
        \\    "vocab": {"[UNK]": 0}
        \\  },
        \\  "normalizer": {
        \\    "type": "SomeUnknownNormalizer"
        \\  }
        \\}
    ;

    var config = try loadConfig(allocator, json_content);
    defer cleanupConfig(&config);

    // Unsupported normalizers return null (skip normalization)
    try std.testing.expect(config.normalizer_impl == null);
}

test "helper getStringField" {
    const json_content =
        \\{"name": "test", "count": 42}
    ;
    const parsed = std.json.parseFromSlice(std.json.Value, std.testing.allocator, json_content, .{}) catch unreachable;
    defer parsed.deinit();

    const obj = parsed.value.object;

    // String field
    try std.testing.expectEqualStrings("test", getStringField(obj, "name").?);

    // Non-string field returns null
    try std.testing.expect(getStringField(obj, "count") == null);

    // Missing field returns null
    try std.testing.expect(getStringField(obj, "missing") == null);
}

test "helper getBoolField" {
    const json_content =
        \\{"enabled": true, "disabled": false, "name": "test"}
    ;
    const parsed = std.json.parseFromSlice(std.json.Value, std.testing.allocator, json_content, .{}) catch unreachable;
    defer parsed.deinit();

    const obj = parsed.value.object;

    // Bool fields
    try std.testing.expectEqual(@as(?bool, true), getBoolField(obj, "enabled"));
    try std.testing.expectEqual(@as(?bool, false), getBoolField(obj, "disabled"));

    // Non-bool field returns null
    try std.testing.expect(getBoolField(obj, "name") == null);

    // Missing field returns null
    try std.testing.expect(getBoolField(obj, "missing") == null);
}

test "isPunctuation" {
    // Common punctuation
    try std.testing.expect(isPunctuation('.'));
    try std.testing.expect(isPunctuation(','));
    try std.testing.expect(isPunctuation('!'));
    try std.testing.expect(isPunctuation('?'));
    try std.testing.expect(isPunctuation(';'));
    try std.testing.expect(isPunctuation(':'));
    try std.testing.expect(isPunctuation('('));
    try std.testing.expect(isPunctuation(')'));
    try std.testing.expect(isPunctuation('['));
    try std.testing.expect(isPunctuation(']'));

    // Not punctuation
    try std.testing.expect(!isPunctuation('a'));
    try std.testing.expect(!isPunctuation('Z'));
    try std.testing.expect(!isPunctuation('0'));
    try std.testing.expect(!isPunctuation(' '));
}
