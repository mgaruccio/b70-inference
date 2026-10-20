#!/usr/bin/env python3
"""Reuse the current SYCL validator; require decoder + head + injection per owner."""
from collections import defaultdict
import gzip
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
SYCL = HERE.parent / '20261019-lilicorr-b70-sycl-profile'
_CURRENT = runpy.run_path(str(SYCL / 'analyze-profile.py'))
CURRENT = _CURRENT['request_sycl'].__globals__
BASE, MARKER = CURRENT['BASE'], CURRENT['MARKER']
_BASE_REQUEST = CURRENT['request_sycl']


class RetainedPath(type(Path())):
    """Read the same retained stream from raw/gzip without staging alternate logs."""
    def read_text(self, *args, **kwargs):
        if self.name not in ('server.stderr.log', 'server.stdout.log'):
            return super().read_text(*args, **kwargs)
        packed = self.with_name(self.name + '.gz')
        if not packed.is_file():
            return super().read_text(*args, **kwargs)
        data = gzip.decompress(packed.read_bytes())
        if self.is_file():
            assert self.read_bytes() == data, 'retained log representations differ'
        return data.decode(kwargs.get('encoding') or 'utf-8', kwargs.get('errors') or 'strict')


# These are fresh private analyzer namespaces; never change completed modules/files.
CURRENT['Path'] = RetainedPath
BASE['analyze'].__globals__['Path'] = RetainedPath


def request_sycl(request, phase_request):
    report = _BASE_REQUEST(request, phase_request)  # All original schema/ownership/overlap checks.
    bundles = CURRENT['_graph_bundles'](CURRENT['parse_sycl'](request['lines'], request['first_line']))
    expected = CURRENT['_scheduler_owners'](phase_request['events'])
    owners = defaultdict(list)
    for bundle in bundles:
        assert bundle['summary']['marker_name'] == 'lilicorr.context', 'not explicit context scope'
        owner = CURRENT['_phase_parent'](bundle, phase_request['events'])
        owners[owner['ubatch_line']].append(bundle)
    assert set(owners) == set(expected), 'missing context-scoped injection/scheduler owner'
    prompt = phase_request['counters']['prompt']
    assert prompt in (55, 8270), 'unexpected measured prompt'
    rounds, injections = [], []
    for line, (ubatch, parent) in sorted(expected.items(), key=lambda item: item[1][0]['begin_us']):
        segments = owners[line]
        observed = sum(b['summary']['graph_nodes'] for b in segments)
        covered = sum(b['summary']['covered_nodes'] for b in segments)
        record = dict(ubatch_line=line, scheduler_nodes=ubatch['nodes'],
                      observed_backend_nodes=observed, covered_nodes=covered,
                      backend_metadata_nodes=observed - covered,
                      uninstrumented_scheduler_nodes=ubatch['nodes'] - observed,
                      graphs=[b['graph'] for b in segments], graph_count=len(segments))
        if parent['stage'] == 'process_chunk':
            assert ubatch['nodes'] == 59, 'injection graph contract changed'
            assert covered > 0, 'injection has no timed operations'
            injections.append(record)
            continue
        assert ubatch['nodes'] == 826, 'full draft graph contract changed'
        projections = [(b, op) for b in segments for op in b['ops']
                       if op['first_op'] == 'MUL_MAT'
                       and op['src0_name'] == 'blk.0.attn_q.weight'
                       and op['src0_type'].lower() == 'bf16']
        assert len(projections) == 1, 'decoder bf16 projection must occur exactly once per round'
        decoder, projection = projections[0]
        heads = [b for b in segments if b['summary']['graph_nodes'] == 167
                 and any(op['src0_name'].startswith('lilicorr.') for op in b['ops'])]
        assert len(heads) == 1, 'missing/duplicate original 167-node head segment'
        assert heads[0]['graph'] != decoder['graph'], 'projection is not in an additional decoder segment'
        assert observed > 167, 'context scope did not expand beyond head-only coverage'
        record.update(round=len(rounds), head_graph=heads[0]['graph'],
                      decoder_graph=decoder['graph'], projection_line=projection['line'],
                      projection_group=projection['group'], projection_count=1)
        rounds.append(record)
    assert len(rounds) == (43 if prompt == 55 else 45), 'missing/extra measured draft round'
    assert injections, 'missing injection owners'
    report['context_scope'] = dict(label='lilicorr.context', decoder_projection_count=len(rounds),
        draft_rounds=len(rounds), injection_owners=len(injections), first_round=rounds[0],
        body_rounds=rounds[1:], injections=injections,
        interpretation='Host wall only (including internal waits); backend metadata and CPU/uninstrumented '
                       'scheduler nodes are not GPU event time. Logging wall is separate.')
    return report


CURRENT['request_sycl'] = request_sycl
for _name in ('event', 'parse_sycl', '_graph_bundles', '_phase_parent', '_scheduler_owners',
              '_unattributed_sycl', '_startup_sycl', '_nonmeasured_sycl', 'analyze'):
    globals()[_name] = CURRENT[_name]
