load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

py_library(
    name = "lars",
    srcs = ["lars.py"],
    srcs_version = "PY3",
)

py_test(
    name = "lars_test",
    size = "medium",
    srcs = ["lars_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":lars"],
)

py_library(
    name = "optimizer_utils",
    srcs = ["optimizer_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":lars",
        ":shampoo",
        ":yogi",
    ],
)

py_test(
    name = "optimizer_utils_test",
    size = "small",
    srcs = ["optimizer_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":optimizer_utils"],
)

py_library(
    name = "shampoo",
    srcs = ["shampoo.py"],
    srcs_version = "PY3",
)

py_test(
    name = "shampoo_test",
    size = "medium",
    srcs = ["shampoo_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":shampoo"],
)

py_library(
    name = "yogi",
    srcs = ["yogi.py"],
    srcs_version = "PY3",
)

py_test(
    name = "yogi_test",
    size = "medium",
    srcs = ["yogi_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":yogi"],
)
