"""Retain visibly unaligned controls for the no-PCA candidate."""
import json,numpy as np
from explore import OUT,render
if __name__=='__main__':
    with np.load(OUT/'panels/pcaNone_initrandom_lr200_perp30.npz') as data:xy=data['native_xy'];labels=data['labels']
    reference=json.loads((OUT/'reference_audit.json').read_text())
    render(xy,labels,'nopca_legacy_unoriented',reference,orient=False)
    render(xy,labels,'nopca_legacy_native_coordinates',reference,orient=False,scale=False)
