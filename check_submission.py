"""Validate a submission against the brief's stated format rules."""
import sys
import numpy as np
import pandas as pd

sub = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else 'submission.csv')
test = pd.read_csv(sys.argv[2] if len(sys.argv) > 2 else 'test.csv')
cols = ['id'] + ['rel%02d' % i for i in range(1, 21)]
ok = True


def chk(cond, msg):
    global ok
    print(('  ok   ' if cond else '  FAIL ') + msg)
    ok = ok and bool(cond)


chk(list(sub.columns) == cols, 'exactly the 21 required columns, in order')
chk(len(sub) == len(test), 'one row per test id (%d)' % len(test))
chk(not sub['id'].duplicated().any(), 'no duplicate ids')
chk(set(sub['id']) == set(test['id']), 'id set matches test.csv exactly')
chk(not sub['id'].isna().any() and (sub['id'].astype(str).str.len() > 0).all(), 'no blank ids')
v = sub[cols[1:]].to_numpy(dtype=float)
chk(np.isfinite(v).all(), 'every score finite (no NaN, no inf)')
chk((v >= 0).all() and (v <= 1).all(), 'every score within [0, 1]  (min %.4f max %.4f)' % (v.min(), v.max()))
ties = (v == v.max(1, keepdims=True)).sum(1)
print('  info  rows whose top score is tied: %d / %d' % ((ties > 1).sum(), len(v)))
print('  info  mean distinct scores per row: %.1f' % np.mean([len(set(r)) for r in v]))
print('PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
