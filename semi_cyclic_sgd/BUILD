load("@rules_python//python:defs.bzl", "py_binary", "py_library")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])

py_binary(
    name = "cyclic_bag_log_reg",
    srcs = ["cyclic_bag_log_reg.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":sentiment_util"],
)

py_binary(
    name = "preprocess_sentiment140",
    srcs = ["preprocess_sentiment140.py"],
    python_version = "PY3",
    srcs_version = "PY3",
)

py_library(
    name = "sentiment_util",
    srcs = ["sentiment_util.py"],
    srcs_version = "PY3",
)
