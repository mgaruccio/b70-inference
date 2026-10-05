#!/usr/bin/env python3
"""Resume this existing run: reuse measured baseline and all374 completed captures."""
import importlib.util
import json
from pathlib import Path
import sys
root=Path.home()/'qwen-mtp-run'
code=Path.home()/'qwen-mtp-code'
spec=importlib.util.spec_from_file_location('full_workflow',code/'run-pilot.py')
workflow=importlib.util.module_from_spec(spec);spec.loader.exec_module(workflow)
train=root/'capture-train/train'
assert len(list(train.glob('*.pt')))==374
baseline=json.loads((root/'stock-baseline.json').read_text())
for depth in (0,4,8):
    stored=json.loads((root/('stock-depth'+str(depth))/'summary.json').read_text())
    assert stored==baseline[str(depth)] and stored['status']=='completed' and stored['requests']==64
original_evaluate=workflow.evaluate
original_capture=workflow.capture
original_archive=workflow.archive
def evaluate(label,depth,split,weights=None):
    if label=='stock-depth'+str(depth) and split=='dev' and weights is None:
        print('REUSE_MEASURED_STOCK_BASELINE='+label,flush=True)
        return baseline[str(depth)]
    return original_evaluate(label,depth,split,weights)
def capture():
    print('REUSE_ALL374_TRAINING_CAPTURES_GENERATE_ALL64_DEV',flush=True)
    original_capture(splits=('dev',),label='teacher-capture-dev-resume2')
    assert len(list(train.glob('*.pt')))==374
    assert len(list((root/'capture-dev/heldout').glob('*.pt')))==64
def archive(phase):
    if phase=='baseline':
        print('REUSE_ALREADY_ARCHIVED_BASELINE_NO_DUPLICATE_GENERATION',flush=True)
        return
    return original_archive(phase)
workflow.evaluate=evaluate
workflow.capture=capture
workflow.archive=archive
workflow.main()
