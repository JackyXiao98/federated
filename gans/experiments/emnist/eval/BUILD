load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//gans/experiments/emnist:__subpackages__"],
)

licenses(["notice"])

py_library(
    name = "emnist_eval_util",
    srcs = ["emnist_eval_util.py"],
    srcs_version = "PY3",
)

py_test(
    name = "emnist_eval_util_test",
    srcs = ["emnist_eval_util_test.py"],
    local = True,
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":emnist_eval_util",
        "//gans/experiments/emnist:emnist_data_utils",
        "//gans/experiments/emnist/classifier:emnist_classifier_model",
    ],
)
