//! Basic tokenization example
//!
//! Demonstrates loading a tokenizer from tokenizer.json and encoding text.

const std = @import("std");
const tokenizer = @import("tokenizer");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        std.debug.print("Usage: {s} <tokenizer.json> [text]\n", .{args[0]});
        std.debug.print("\nExample:\n", .{});
        std.debug.print("  {s} path/to/tokenizer.json \"Hello, world!\"\n", .{args[0]});
        return;
    }

    const tokenizer_path = args[1];
    const text = if (args.len > 2) args[2] else "Hello, world!";

    std.debug.print("Loading tokenizer from: {s}\n", .{tokenizer_path});

    var tok = try tokenizer.Tokenizer.fromFile(allocator, tokenizer_path);
    defer tok.deinit();

    std.debug.print("Tokenizing: \"{s}\"\n\n", .{text});

    var encoding = try tok.encode(text, true);
    defer encoding.deinit();

    std.debug.print("Tokens ({d}):\n", .{encoding.len()});
    for (encoding.getIds(), encoding.getTokens(), 0..) |id, token_str, i| {
        std.debug.print("  [{d:4}] {d:6} = \"{s}\"\n", .{ i, id, token_str });
    }

    std.debug.print("\nIDs: ", .{});
    for (encoding.getIds()) |id| {
        std.debug.print("{d} ", .{id});
    }
    std.debug.print("\n", .{});
}
