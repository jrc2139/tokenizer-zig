const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Library module - export for external use
    const tokenizer_mod = b.addModule("tokenizer", .{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Tests
    const lib_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/lib.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_lib_tests = b.addRunArtifact(lib_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_tests.step);

    // Example
    const example = b.addExecutable(.{
        .name = "basic_tokenize",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/basic_tokenize.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "tokenizer", .module = tokenizer_mod },
            },
        }),
    });
    b.installArtifact(example);

    const run_example = b.addRunArtifact(example);
    if (b.args) |args| {
        run_example.addArgs(args);
    }
    const example_step = b.step("example", "Run the basic tokenize example");
    example_step.dependOn(&run_example.step);

    // Check step (fast compile check)
    const check = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/lib.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const check_step = b.step("check", "Check if code compiles");
    check_step.dependOn(&check.step);
}
