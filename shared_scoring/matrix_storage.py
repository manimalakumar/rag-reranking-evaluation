"""Reference existing vectors instead of storing a second matrix."""
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np


def npy_sha256(matrix):
    """Hash the unchanged NPY representation without writing a second file."""
    header=io.BytesIO()
    np.lib.format.write_array_header_1_0(header,dict(descr=np.lib.format.dtype_to_descr(matrix.dtype),fortran_order=False,shape=matrix.shape))
    digest=hashlib.sha256(header.getvalue())
    for start in range(0,len(matrix),1024):
        digest.update(np.ascontiguousarray(matrix[start:start+1024]).tobytes())
    return digest.hexdigest()


def reference_path(path):
    return Path(path).with_suffix('.reference.json')


def write_reference(path, source, matrix, indices=None):
    path=Path(path)
    data=dict(source_file=os.path.relpath(Path(source).resolve(),path.parent.resolve()),
              rows=len(matrix),dimensions=matrix.shape[1],dtype=matrix.dtype.str,npy_sha256=npy_sha256(matrix))
    if indices is not None:
        data['indices']=list(map(int,indices))
    target=reference_path(path)
    temporary=target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n')
    temporary.replace(target)
    return data['npy_sha256']


def load_matrix(path):
    path=Path(path)
    if path.exists():
        return np.load(path,mmap_mode='r')
    data=json.loads(reference_path(path).read_text())
    source=np.load((path.parent/data['source_file']).resolve(),mmap_mode='r')
    matrix=source[data['indices']] if 'indices' in data else source[:data['rows']]
    if matrix.shape!=(data['rows'],data['dimensions']) or matrix.dtype.str!=data['dtype']:
        raise ValueError('Referenced matrix shape or dtype changed.')
    return matrix
