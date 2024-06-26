load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

# Enables linear compression aggregators in Federated Learning as a means to achieve optimal privacy-accuracy-communication tradeoffs.

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])

py_library(
    name = "count_sketching",
    srcs = ["count_sketching.py"],
    srcs_version = "PY3",
    deps = [":count_sketching_utils"],
)

py_library(
    name = "ddp_compression",
    srcs = ["ddp_compression.py"],
    srcs_version = "PY3",
    deps = [
        ":count_sketching",
        ":count_sketching_utils",
    ],
)

py_test(
    name = "ddp_compression_test",
    size = "large",
    srcs = ["ddp_compression_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":ddp_compression"],
)

py_test(
    name = "count_sketching_test",
    srcs = ["count_sketching_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":count_sketching",
        ":count_sketching_utils",
    ],
)

py_library(
    name = "count_sketching_utils",
    srcs = ["count_sketching_utils.py"],
    srcs_version = "PY3",
)

py_binary(
    name = "dme_merge_and_plot",
    srcs = ["dme_merge_and_plot.py"],
)

py_test(
    name = "count_sketching_utils_test",
    srcs = ["count_sketching_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":count_sketching_utils"],
)

py_binary(
    name = "fl_compression_run",
    srcs = ["fl_compression_run.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":count_sketching",
        ":count_sketching_utils",
        ":ddp_compression",
        "//utils:task_utils",
        "//utils:training_utils",
        "//utils:utils_impl",
        "//utils/optimizers:optimizer_utils",
    ],
)

py_library(
    name = "subsampled_random_hadamard",
    srcs = ["subsampled_random_hadamard.py"],
    srcs_version = "PY3",
    deps = ["//distributed_dp:compression_utils"],
)

py_test(
    name = "subsampled_random_hadamard_test",
    srcs = ["subsampled_random_hadamard_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":subsampled_random_hadamard"],
)

py_binary(
    name = "dme_run",
    srcs = ["dme_run.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":count_sketching_utils",
        "//distributed_dp:accounting_utils",
        "//distributed_dp:ddpquery_utils",
        "//distributed_dp:dme_utils",
    ],
)
