#!/usr/bin/env python3
"""Pure checks for the seeded N=9 native output-equivalence proof."""
import hashlib
import re

MARKER = 'BOLT_N9_OUTPUT_HEX='
EXPECTED_BYTES = 17856
TARGET_ROUTE = 'router_columns8-m496-l0-n9'
SEED = '0x5A17 + chunk start'
FILTER = r'^router_columns8=1,router_single2=1,weight=blk\.48\.ffn_gate_inp\.weight,layout=0,type_a=f32,type_b=f32,m=496,n=9,'

SOURCE_REPLACEMENTS = (
    (
        b'thread_local std::default_random_engine gen(std::random_device{}());',
        b'std::default_random_engine gen(static_cast<unsigned int>(0x5A17u + start));',
    ),
    (
        b'''            double err = ud->tc->err(f1.data(), f2.data(), f1.size());
            if (err > ud->tc->max_err(ud->backend1)) {''',
        b'''            if (strcmp(bn1, "SYCL0") == 0 && strcmp(bn2, "CPU") == 0 &&
                    strcmp(t1->name, "router_columns8-m496-l0-n9") == 0 &&
                    t1->type == GGML_TYPE_F32 && ggml_is_contiguous(t1) &&
                    t1->ne[0] == 496 && t1->ne[1] == 9 && t1->ne[2] == 1 && t1->ne[3] == 1 &&
                    ggml_nbytes(t1) == 17856) {
                std::vector<uint8_t> device_output(ggml_nbytes(t1));
                ggml_backend_tensor_get(t1, device_output.data(), 0, device_output.size());
                printf("\nBOLT_N9_OUTPUT_HEX=");
                for (uint8_t byte : device_output) {
                    printf("%02x", byte);
                }
                printf("\\n");
            }

            double err = ud->tc->err(f1.data(), f2.data(), f1.size());
            if (err > ud->tc->max_err(ud->backend1)) {''',
    ),
)


def expected_test_source(source):
    """Return the only permitted test source after applying the seeded patch."""
    result = source
    for before, after in SOURCE_REPLACEMENTS:
        assert result.count(before) == 1
        result = result.replace(before, after)
    return result


def marker_bytes(raw):
    if isinstance(raw, bytes):
        raw = raw.decode('utf-8')
    lines = [line for line in raw.splitlines() if line.startswith(MARKER)]
    assert len(lines) == 1, ('N=9 output marker count', len(lines))
    encoded = lines[0][len(MARKER):]
    assert re.fullmatch(r'[0-9a-f]{%d}' % (2 * EXPECTED_BYTES), encoded)
    output = bytes.fromhex(encoded)
    assert len(output) == EXPECTED_BYTES
    return output


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def check_pair(parent_raw, candidate_raw):
    parent = marker_bytes(parent_raw)
    candidate = marker_bytes(candidate_raw)
    assert parent == candidate, 'N=9 device output bytes differ'
    return dict(baseline_equivalent=True, marker=MARKER, byte_length=EXPECTED_BYTES,
                route=TARGET_ROUTE, seed=SEED, filter=FILTER)


def make_proof(parent_raw, candidate_raw, backend_hashes, test_binary, test_source,
               controller_sha256, raw_hashes):
    result = check_pair(parent_raw, candidate_raw)
    result.update(backend_hashes=backend_hashes, test_binary=test_binary,
                  test_source=test_source, controller_sha256=controller_sha256,
                  raw_sha256=raw_hashes)
    return result


def check_proof(proof, parent_raw, candidate_raw, backend_hashes, test_binary,
                test_source, controller_sha256, raw_hashes):
    expected = check_pair(parent_raw, candidate_raw)
    assert proof.get('baseline_equivalent') is True
    for key, value in expected.items():
        assert proof.get(key) == value, (key, proof.get(key))
    assert proof.get('backend_hashes') == backend_hashes
    assert proof.get('test_binary') == test_binary
    assert proof.get('test_source') == test_source
    assert proof.get('controller_sha256') == controller_sha256
    assert proof.get('raw_sha256') == raw_hashes
    return proof
