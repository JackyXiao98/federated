load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//triehh:__subpackages__"],
)

licenses(["notice"])

py_library(
    name = "heavy_hitters_utils",
    srcs = ["heavy_hitters_utils.py"],
    srcs_version = "PY3",
)

py_library(
    name = "heavy_hitters_testcase",
    testonly = True,
    srcs = ["heavy_hitters_testcase.py"],
    srcs_version = "PY3",
)

py_test(
    name = "heavy_hitters_utils_test",
    srcs = ["heavy_hitters_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "manual",
    ],
    deps = [":heavy_hitters_utils"],
)
