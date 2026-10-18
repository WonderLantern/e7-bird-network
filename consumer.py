from pathlib import Path
import ctypes as C,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
class Box(C.Structure):
    _fields_ = [('x', C.c_float), ('y', C.c_float), ('w', C.c_float), ('h', C.c_float), ('score', C.c_uint16), ('cls', C.c_uint16)]

class Report(C.Structure):
    _fields_ = [('visited', C.c_uint32), ('accepted', C.c_uint32), ('outputs', C.c_uint32), ('error_ordinal', C.c_uint32), ('error_field', C.c_uint32), ('initial', C.c_uint16 * 24), ('ordinals', C.c_uint32 * 32), ('candidates', Box * 32), ('result', C.c_uint16 * 6 * 16)]

def pack_heads(fine, coarse):
    assert fine.shape == (22, 16, 24) and coarse.shape == (33, 8, 12)
    with np.errstate(over='ignore'):
        f = np.asarray(fine, dtype='<f2')
        c = np.asarray(coarse, dtype='<f2')
    padded = np.zeros((33, 8, 16), dtype='<f2')
    padded[:, :, :12] = c
    return f.tobytes() + padded.tobytes()

class Consumer:
    def __init__(self):
        self.manifest=json.loads((ROOT/'manifest.json').read_text())
        self.dll=C.CDLL(str(ROOT/'wide-reference.dll'))
        assert C.sizeof(Box)==20 and C.sizeof(Report)==1028
        self.dll.reference_mode.restype=C.c_int;assert self.dll.reference_mode()==1
        self.dll.reference_consume.argtypes=[C.POINTER(C.c_uint8),C.c_size_t,C.POINTER(C.c_uint8),C.c_size_t,C.POINTER(Report)]
        self.dll.reference_consume.restype=C.c_int
        ipu=bytearray(0x80000)
        for row in self.manifest['consumer_tables']:
            raw=(ROOT/row['file']).read_bytes();assert hashlib.sha256(raw).hexdigest()==row['sha256']
            ipu[row['offset']:row['offset']+len(raw)]=raw
        self.ipu=(C.c_uint8*len(ipu)).from_buffer_copy(ipu)

    def consume(self, buffer):
        raw = (C.c_uint8 * len(buffer)).from_buffer_copy(buffer)
        report = Report()
        status = self.dll.reference_consume(raw, len(buffer), self.ipu, len(self.ipu), C.byref(report))
        return dict(status=status, records=[list(r) for r in report.result[:report.outputs]], error_field=report.error_field, error_ordinal=report.error_ordinal, visited=report.visited)
    def select(self, records, mode):
        if mode == 'raw':
            return records
        cutoff = self.manifest['score_conditions'][str(mode)]
        return [r for r in records if r[4] >= cutoff[r[5]]]
