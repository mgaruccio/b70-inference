#!/usr/bin/env python3
"""Emit (do not apply) the M5 gate/up catalog experiment's one-file unified diff.

Usage: python patch-catalog.py /path/to/oneDNN > catalog.patch
Then, separately: git -C /path/to/oneDNN apply --check /path/to/catalog.patch
                  git -C /path/to/oneDNN apply /path/to/catalog.patch

Requires the exact pinned Git HEAD and an unchanged jit.hpp. No downloads,
source/archive writes, or other file changes are performed by this script.
Repeated application and unexpected source markers fail closed.

Fresh upstream API inspection (2026-09-13):
https://raw.githubusercontent.com/uxlfoundation/oneDNN/80afa71049cd69a3df32adcccb623b12cd7baa22/src/gpu/intel/gemm/jit.hpp
https://raw.githubusercontent.com/uxlfoundation/oneDNN/80afa71049cd69a3df32adcccb623b12cd7baa22/src/gpu/intel/gemm/jit/gen_kernel.hpp
https://raw.githubusercontent.com/uxlfoundation/oneDNN/80afa71049cd69a3df32adcccb623b12cd7baa22/src/gpu/intel/gemm/jit/pd.cpp

G128 is checked on the u4 operand's actual scale attributes, not zero-point
groups. Descriptor A/B attributes precede swap_ab_; M/N accept either order.
All nonmatching descriptors and all original validation checks stay unchanged.
A forced entry that reaches this selector cannot fall back on failure: the
primitive descriptor creation iterator does not catch std::runtime_error.

Use a fresh probe process for each env value: native and oneDNN primitive caches
are intentionally unchanged and do not key on this experiment-only variable.
The probe must require a successful chosen-identity log; a rejection before
this catalog selector is not evidence that a forced catalog entry ran.
"""

import argparse
import difflib
from pathlib import Path
import subprocess
import sys


REVISION = "80afa71049cd69a3df32adcccb623b12cd7baa22"
SOURCE_PATH = "src/gpu/intel/gemm/jit.hpp"
ENV_NAME = "B70_M5_GATEUP_CATALOG_INDEX"

SELECT_MARKER = """            auto entries = kernel_desc_.select_kernel(arch_, stepping,
                    dev_info_->eu_count(), has_systolic, is_integrated, mode,
                    problem, alpha(), beta(), m, n, d->k(), lda, ldb, d->ldc(),
                    d->batch());
"""

CATALOG_SELECTION = r"""
            // B70_M5_GATEUP_CATALOG_INDEX: isolated, creation-only experiment.
            const auto &b70_a_scales = attr()->scales_.get(DNNL_ARG_A);
            const auto &b70_b_scales = attr()->scales_.get(DNNL_ARG_B);
            const bool b70_gateup = d->batch() == 1 && d->k() == 5120
                    && ((m == 5 && n == 34816) || (m == 34816 && n == 5))
                    && d->c_type() == f16
                    && ((d->a_type() == u4 && d->b_type() == f16
                                && !b70_a_scales.has_default_groups()
                                && b70_a_scales.get_group(0) == 128
                                && b70_a_scales.get_group(1) == 1)
                            || (d->a_type() == f16 && d->b_type() == u4
                                    && !b70_b_scales.has_default_groups()
                                    && b70_b_scales.get_group(0) == 1
                                    && b70_b_scales.get_group(1) == 128));
            int b70_forced_index = -1;
            int b70_attempt_index = -1;
            if (b70_gateup) {
                const char *b70_value = std::getenv("B70_M5_GATEUP_CATALOG_INDEX");
                if (b70_value) {
                    if (b70_value[0] == '-' && b70_value[1] == '1'
                            && b70_value[2] == '\0') {
                        // Explicit auto, identical to the unset default.
                    } else if (b70_value[0] >= '0' && b70_value[0] <= '9'
                            && b70_value[1] == '\0') {
                        b70_forced_index = b70_value[0] - '0';
                    } else if (b70_value[0] == '1'
                            && b70_value[1] >= '0' && b70_value[1] <= '2'
                            && b70_value[2] == '\0') {
                        b70_forced_index = 10 + b70_value[1] - '0';
                    } else {
                        throw std::runtime_error(
                                "B70_M5_GATEUP_CATALOG_INDEX: expected -1 or 0 through 12");
                    }
                }
                std::fprintf(stderr,
                        "b70_gemm_catalog,create,m=%lld,n=%lld,k=5120,batch=1,"
                        "group_k=128,requested=%d,entries=%zu\n",
                        static_cast<long long>(m), static_cast<long long>(n),
                        b70_forced_index, entries.size());
                for (size_t b70_i = 0; b70_i < entries.size() && b70_i < 13;
                        ++b70_i) {
                    std::fprintf(stderr,
                            "b70_gemm_catalog,candidate,index=%zu,identity=%s\n",
                            b70_i, entries[b70_i]->str().c_str());
                }
                if (b70_forced_index >= 0) {
                    if (static_cast<size_t>(b70_forced_index) >= entries.size())
                        throw std::runtime_error(
                                "B70_M5_GATEUP_CATALOG_INDEX: index outside returned catalog; "
                                "refusing implementation fallback");
                    const auto *b70_entry = entries[b70_forced_index];
                    entries = {b70_entry};
                }
            }
"""

SUCCESS_LOG = r"""                        if (b70_gateup)
                            std::fprintf(stderr,
                                    "b70_gemm_catalog,selected,mode=%s,index=%d,identity=%s\n",
                                    b70_forced_index < 0 ? "auto" : "forced",
                                    b70_forced_index < 0 ? b70_attempt_index
                                                         : b70_forced_index,
                                    kernel_desc_.entry().str().c_str());
"""

FAILURE_GUARD = """            if (b70_gateup && b70_forced_index >= 0 && !kernel_success)
                throw std::runtime_error(
                        "B70_M5_GATEUP_CATALOG_INDEX: forced entry failed original "
                        "validation or kernel creation; refusing implementation fallback");

"""


def replace_once(source: str, marker: str, replacement: str) -> str:
    count = source.count(marker)
    if count != 1:
        raise ValueError(f"expected one exact source marker, found {count}: {marker!r}")
    return source.replace(marker, replacement, 1)


def patched_source(source: str) -> str:
    if ENV_NAME in source or "b70_gemm_catalog" in source:
        raise ValueError("catalog experiment already present; refusing repeated application")
    include_marker = "#include <assert.h>\n"
    source = replace_once(
        source, include_marker,
        include_marker + "#include <cstdlib>\n#include <cstdio>\n#include <stdexcept>\n",
    )
    source = replace_once(source, SELECT_MARKER, SELECT_MARKER + CATALOG_SELECTION)
    loop_marker = "            for (auto &entry : entries) {\n"
    source = replace_once(
        source, loop_marker, loop_marker + "                ++b70_attempt_index;\n"
    )
    success_marker = "                        kernel_success = true;\n"
    source = replace_once(source, success_marker, success_marker + SUCCESS_LOG)
    failure_marker = """            VDISPATCH_GEMM(
                    kernel_success, "matching kernel not found in catalog");
"""
    return replace_once(source, failure_marker, FAILURE_GUARD + failure_marker)


def git(source_dir: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(source_dir), *args], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ).stdout


def read_pinned_source(source_dir: Path) -> str:
    if Path(git(source_dir, "rev-parse", "--show-toplevel").strip()).resolve() != source_dir:
        raise ValueError("source path must be the oneDNN Git checkout root")
    head = git(source_dir, "rev-parse", "HEAD").strip()
    if head != REVISION:
        raise ValueError(f"expected oneDNN HEAD {REVISION}, got {head}")
    source_file = source_dir / SOURCE_PATH
    if source_file.resolve() != source_file:
        raise ValueError("refusing a symlinked jit.hpp or source path component")
    source = source_file.read_bytes().decode("utf-8")
    if ENV_NAME in source or "b70_gemm_catalog" in source:
        raise ValueError("catalog experiment already present; refusing repeated application")
    if source != git(source_dir, "show", f"{REVISION}:{SOURCE_PATH}"):
        raise ValueError("jit.hpp differs from the pinned Git revision; refusing to patch")
    return source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path, help="pinned oneDNN Git checkout root")
    args = parser.parse_args()
    try:
        source = read_pinned_source(args.source_dir.resolve(strict=True))
        patched = patched_source(source)
    except (OSError, UnicodeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"patch-catalog: {exc}", file=sys.stderr)
        return 1
    sys.stdout.writelines(difflib.unified_diff(
        source.splitlines(keepends=True), patched.splitlines(keepends=True),
        fromfile=f"a/{SOURCE_PATH}", tofile=f"b/{SOURCE_PATH}", n=3,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
