import struct, sys
f = open(sys.argv[1], 'rb')
magic, ver = f.read(4), struct.unpack('<I', f.read(4))[0]
nt, nkv = struct.unpack('<QQ', f.read(16))
def rs():
    n = struct.unpack('<Q', f.read(8))[0]; return f.read(n).decode('utf-8', 'replace')
SZ = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
def rv(t):
    if t == 8: return rs()
    if t == 9:
        it = struct.unpack('<I', f.read(4))[0]; n = struct.unpack('<Q', f.read(8))[0]
        if it == 8: return [rs() for _ in range(n)] if n < 64 else (f'<{n} strings>', [rs() for _ in range(n)][:0])
        fmt = SZ[it]; sz = struct.calcsize(fmt); data = f.read(sz * n)
        vals = list(struct.unpack('<' + fmt * n, data))
        return vals if n <= 64 else f'<array len {n}: first {vals[:8]}>'
    fmt = SZ[t]; return struct.unpack('<' + fmt, f.read(struct.calcsize(fmt)))[0]
print('version', ver, 'tensors', nt, 'kv', nkv)
for _ in range(nkv):
    k = rs(); t = struct.unpack('<I', f.read(4))[0]; v = rv(t)
    if 'tokenizer.ggml' in k and isinstance(v, (list, tuple)) and len(str(v)) > 300: v = str(v)[:120]
    print(k, '=', v)
