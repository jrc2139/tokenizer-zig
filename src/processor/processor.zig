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
