load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//fedopt_guide:__subpackages__"],
)

licenses(["notice"])

py_library(
    name = "centralized_training_loop",
    srcs = ["centralized_training_loop.py"],
    srcs_version = "PY3",
    deps = ["//utils:utils_impl"],
)

py_test(
    name = "centralized_training_loop_test",
    srcs = ["centralized_training_loop_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":centralized_training_loop"],
)

py_library(
    name = "training_loop",
    srcs = ["training_loop.py"],
    srcs_version = "PY3",
    deps = ["//utils:utils_impl"],
)

py_test(
    name = "training_loop_test",
    size = "large",
    srcs = ["training_loop_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = ["//fedopt_guide:training_loop"],
)
