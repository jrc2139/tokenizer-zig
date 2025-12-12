//! PostProcessor interface for post-processing tokenization output
//!
//! Post-processors add special tokens like [CLS], [SEP], etc.
//! and handle sequence pair encoding.

const std = @import("std");
const Encoding = @import("../encoding.zig").Encoding;

/// Function pointer types
pub const ProcessFn = *const fn (ctx: *anyopaque, encoding: *Encoding) anyerror!void;
pub const ProcessPairFn = *const fn (ctx: *anyopaque, encoding: *Encoding, pair: *Encoding) anyerror!void;
pub const DeinitFn = *const fn (ctx: *anyopaque) void;

/// PostProcessor interface
pub const PostProcessor = struct {
    ptr: *anyopaque,
    process_fn: ProcessFn,
    process_pair_fn: ?ProcessPairFn = null,
    deinit_fn: ?DeinitFn = null,

    const Self = @This();

    pub fn process(self: Self, encoding: *Encoding) !void {
        return self.process_fn(self.ptr, encoding);
    }

    pub fn processPair(self: Self, encoding: *Encoding, pair: *Encoding) !void {
        if (self.process_pair_fn) |f| {
            return f(self.ptr, encoding, pair);
        }
    }

    pub fn deinit(self: *Self) void {
        if (self.deinit_fn) |f| {
            f(self.ptr);
        }
    }
};

/// Special token definition for templates
pub const SpecialToken = struct {
    id: []const u8,
    ids: []const u32,
    tokens: []const []const u8,
};

/// BERT post-processor - adds [CLS] and [SEP] tokens
pub const BertProcessing = struct {
    sep: struct { token: []const u8, id: u32 },
    cls: struct { token: []const u8, id: u32 },

    const Self = @This();

    pub fn init(sep_token: []const u8, sep_id: u32, cls_token: []const u8, cls_id: u32) Self {
        return .{
            .sep = .{ .token = sep_token, .id = sep_id },
            .cls = .{ .token = cls_token, .id = cls_id },
        };
    }

    pub fn postProcessor(self: *Self) PostProcessor {
        return .{
            .ptr = self,
            .process_fn = processImpl,
            .process_pair_fn = processPairImpl,
        };
    }

    fn processImpl(ctx: *anyopaque, encoding: *Encoding) anyerror!void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        _ = self;
        _ = encoding;
        // TODO: Add [CLS] at start and [SEP] at end
    }

    fn processPairImpl(ctx: *anyopaque, encoding: *Encoding, pair: *Encoding) anyerror!void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        _ = self;
        _ = encoding;
        _ = pair;
        // TODO: Handle sequence pairs [CLS] A [SEP] B [SEP]
    }
};

/// Template-based post-processor
pub const TemplateProcessing = struct {
    single: []const TemplatePiece,
    pair: ?[]const TemplatePiece,
    special_tokens: []const SpecialToken,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub const TemplatePiece = union(enum) {
        sequence: struct { id: u8, type_id: u32 },
        special_token: struct { id: []const u8 },
    };

    pub fn postProcessor(self: *Self) PostProcessor {
        return .{
            .ptr = self,
            .process_fn = processImpl,
            .process_pair_fn = processPairImpl,
            .deinit_fn = deinitImpl,
        };
    }

    fn processImpl(ctx: *anyopaque, encoding: *Encoding) anyerror!void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        _ = self;
        _ = encoding;
        // TODO: Apply template
    }

    fn processPairImpl(ctx: *anyopaque, encoding: *Encoding, pair: *Encoding) anyerror!void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        _ = self;
        _ = encoding;
        _ = pair;
        // TODO: Apply pair template
    }

    fn deinitImpl(ctx: *anyopaque) void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        _ = self;
        // Free allocated memory
    }
};

/// RoBERTa post-processor
pub const RobertaProcessing = struct {
    sep: struct { token: []const u8, id: u32 },
    cls: struct { token: []const u8, id: u32 },
    trim_offsets: bool = true,
    add_prefix_space: bool = true,

    const Self = @This();

    pub fn postProcessor(self: *Self) PostProcessor {
        return .{
            .ptr = self,
            .process_fn = processImpl,
            .process_pair_fn = processPairImpl,
        };
    }

    fn processImpl(ctx: *anyopaque, encoding: *Encoding) anyerror!void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        _ = self;
        _ = encoding;
        // TODO: Add <s> at start and </s> at end
    }

    fn processPairImpl(ctx: *anyopaque, encoding: *Encoding, pair: *Encoding) anyerror!void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        _ = self;
        _ = encoding;
        _ = pair;
        // TODO: Handle sequence pairs <s> A </s></s> B </s>
    }
};

// ============================================================================
// Tests
// ============================================================================

test "BertProcessing: init stores tokens" {
    const bp = BertProcessing.init("[SEP]", 102, "[CLS]", 101);

    try std.testing.expectEqualStrings("[SEP]", bp.sep.token);
    try std.testing.expectEqual(@as(u32, 102), bp.sep.id);
    try std.testing.expectEqualStrings("[CLS]", bp.cls.token);
    try std.testing.expectEqual(@as(u32, 101), bp.cls.id);
}

test "BertProcessing: postProcessor interface" {
    var bp = BertProcessing.init("[SEP]", 102, "[CLS]", 101);
    const pp = bp.postProcessor();

    // Verify process_pair_fn is set (not null)
    try std.testing.expect(pp.process_pair_fn != null);
}

test "BertProcessing: process single encoding (stub)" {
    const allocator = std.testing.allocator;
    var bp = BertProcessing.init("[SEP]", 102, "[CLS]", 101);
    const pp = bp.postProcessor();

    var encoding = Encoding.empty(allocator);
    defer encoding.deinit();

    // Currently a stub - should not error
    try pp.process(&encoding);
}

test "BertProcessing: processPair (stub)" {
    const allocator = std.testing.allocator;
    var bp = BertProcessing.init("[SEP]", 102, "[CLS]", 101);
    const pp = bp.postProcessor();

    var encoding1 = Encoding.empty(allocator);
    defer encoding1.deinit();
    var encoding2 = Encoding.empty(allocator);
    defer encoding2.deinit();

    // Currently a stub - should not error
    try pp.processPair(&encoding1, &encoding2);
}

test "BertProcessing: deinit is null (safe)" {
    var bp = BertProcessing.init("[SEP]", 102, "[CLS]", 101);
    var pp = bp.postProcessor();

    // deinit_fn is null, should not crash
    pp.deinit();
}

test "TemplateProcessing: postProcessor interface" {
    const allocator = std.testing.allocator;
    var tp = TemplateProcessing{
        .single = &.{},
        .pair = null,
        .special_tokens = &.{},
        .allocator = allocator,
    };
    const pp = tp.postProcessor();

    // Verify deinit_fn is set (not null)
    try std.testing.expect(pp.deinit_fn != null);
}

test "TemplateProcessing: process (stub)" {
    const allocator = std.testing.allocator;
    var tp = TemplateProcessing{
        .single = &.{},
        .pair = null,
        .special_tokens = &.{},
        .allocator = allocator,
    };
    const pp = tp.postProcessor();

    var encoding = Encoding.empty(allocator);
    defer encoding.deinit();

    // Currently a stub - should not error
    try pp.process(&encoding);
}

test "TemplateProcessing: processPair (stub)" {
    const allocator = std.testing.allocator;
    var tp = TemplateProcessing{
        .single = &.{},
        .pair = null,
        .special_tokens = &.{},
        .allocator = allocator,
    };
    const pp = tp.postProcessor();

    var encoding1 = Encoding.empty(allocator);
    defer encoding1.deinit();
    var encoding2 = Encoding.empty(allocator);
    defer encoding2.deinit();

    // Currently a stub - should not error
    try pp.processPair(&encoding1, &encoding2);
}

test "TemplateProcessing: deinit via interface (stub)" {
    const allocator = std.testing.allocator;
    var tp = TemplateProcessing{
        .single = &.{},
        .pair = null,
        .special_tokens = &.{},
        .allocator = allocator,
    };
    var pp = tp.postProcessor();

    // Currently a stub - should not error
    pp.deinit();
}

test "RobertaProcessing: postProcessor interface" {
    var rp = RobertaProcessing{
        .sep = .{ .token = "</s>", .id = 2 },
        .cls = .{ .token = "<s>", .id = 0 },
    };
    const pp = rp.postProcessor();

    // Verify process_pair_fn is set (not null)
    try std.testing.expect(pp.process_pair_fn != null);
}

test "RobertaProcessing: stores config options" {
    const rp = RobertaProcessing{
        .sep = .{ .token = "</s>", .id = 2 },
        .cls = .{ .token = "<s>", .id = 0 },
        .trim_offsets = false,
        .add_prefix_space = false,
    };

    try std.testing.expectEqual(false, rp.trim_offsets);
    try std.testing.expectEqual(false, rp.add_prefix_space);
}

test "RobertaProcessing: process (stub)" {
    const allocator = std.testing.allocator;
    var rp = RobertaProcessing{
        .sep = .{ .token = "</s>", .id = 2 },
        .cls = .{ .token = "<s>", .id = 0 },
    };
    const pp = rp.postProcessor();

    var encoding = Encoding.empty(allocator);
    defer encoding.deinit();

    // Currently a stub - should not error
    try pp.process(&encoding);
}

test "RobertaProcessing: processPair (stub)" {
    const allocator = std.testing.allocator;
    var rp = RobertaProcessing{
        .sep = .{ .token = "</s>", .id = 2 },
        .cls = .{ .token = "<s>", .id = 0 },
    };
    const pp = rp.postProcessor();

    var encoding1 = Encoding.empty(allocator);
    defer encoding1.deinit();
    var encoding2 = Encoding.empty(allocator);
    defer encoding2.deinit();

    // Currently a stub - should not error
    try pp.processPair(&encoding1, &encoding2);
}

test "RobertaProcessing: deinit is null (safe)" {
    var rp = RobertaProcessing{
        .sep = .{ .token = "</s>", .id = 2 },
        .cls = .{ .token = "<s>", .id = 0 },
    };
    var pp = rp.postProcessor();

    // deinit_fn is null, should not crash
    pp.deinit();
}

test "PostProcessor: processPair with null function is no-op" {
    const allocator = std.testing.allocator;
    // Create a PostProcessor with null process_pair_fn
    const pp = PostProcessor{
        .ptr = undefined,
        .process_fn = struct {
            fn process(_: *anyopaque, _: *Encoding) anyerror!void {}
        }.process,
        .process_pair_fn = null,
    };

    var encoding1 = Encoding.empty(allocator);
    defer encoding1.deinit();
    var encoding2 = Encoding.empty(allocator);
    defer encoding2.deinit();

    // Should be no-op, not error
    try pp.processPair(&encoding1, &encoding2);
}

test "SpecialToken: struct fields" {
    const st = SpecialToken{
        .id = "[CLS]",
        .ids = &.{101},
        .tokens = &.{"[CLS]"},
    };

    try std.testing.expectEqualStrings("[CLS]", st.id);
    try std.testing.expectEqual(@as(usize, 1), st.ids.len);
    try std.testing.expectEqual(@as(u32, 101), st.ids[0]);
}

test "TemplatePiece: sequence variant" {
    const piece = TemplateProcessing.TemplatePiece{ .sequence = .{ .id = 0, .type_id = 0 } };
    switch (piece) {
        .sequence => |s| {
            try std.testing.expectEqual(@as(u8, 0), s.id);
            try std.testing.expectEqual(@as(u32, 0), s.type_id);
        },
        .special_token => unreachable,
    }
}

test "TemplatePiece: special_token variant" {
    const piece = TemplateProcessing.TemplatePiece{ .special_token = .{ .id = "[CLS]" } };
    switch (piece) {
        .sequence => unreachable,
        .special_token => |st| {
            try std.testing.expectEqualStrings("[CLS]", st.id);
        },
    }
}
