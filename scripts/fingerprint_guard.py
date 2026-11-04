"""Freeze owned workers while investigating a transient fingerprint mismatch."""
import datetime
import os
from pathlib import Path
import signal
import time
from common import save, job_processes
from experiment_pipeline import fingerprint


class FingerprintGuard:
    def __init__(self, digest, output, reader=fingerprint):
        self.digest, self.output, self.reader = digest, Path(output), reader
        self.baseline = {}
        observed = reader(details=self.baseline)
        if observed != digest:
            raise RuntimeError('Fingerprint changed before controller initialization')
        save(self.output / 'fingerprint_baseline.json', {'digest': digest, 'components': self.baseline})

    def check(self, processes):
        current = {}
        observed = self.reader(details=current)
        if observed == self.digest:
            return
        # Freeze only this campaign's jobs. Retrieval remains available, and
        # no additional search work proceeds during verification of the code.
        frozen = set()
        for process in processes:
            if process.poll() is None:
                frozen.update(job_processes(process.pid))
        for pid in sorted(frozen, reverse=True):
            try:
                os.kill(pid, signal.SIGSTOP)
            except ProcessLookupError:
                pass
        keys = self.baseline.keys() | current.keys()
        differences = {key: {'expected': self.baseline.get(key), 'observed': current.get(key)}
                       for key in keys if self.baseline.get(key) != current.get(key)}
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        path = self.output / f'fingerprint_incident.{stamp}.json'
        incident = {'expected': self.digest, 'observed': observed, 'differences': differences,
                    'frozen_pids': sorted(frozen), 'status': 'VERIFYING', 'rechecks': []}
        save(path, incident)
        recovered = False
        try:
            # Both complete independent reads must match the original digest.
            # Persistent changes still fail closed; the baseline is never reset.
            for _ in range(2):
                time.sleep(1)
                value = self.reader()
                incident['rechecks'].append(value)
                if value != self.digest:
                    raise RuntimeError(f'Persistent code/config fingerprint change; details: {path}')
            recovered = True
            incident['status'] = 'TRANSIENT_RECOVERED'
            print(f'TRANSIENT fingerprint mismatch verified against original twice; evidence: {path}', flush=True)
        finally:
            incident['status'] = 'TRANSIENT_RECOVERED' if recovered else 'PERSISTENT_CHANGE_ABORTED'
            save(path, incident)
            # Resume before normal controller cleanup can terminate failed jobs.
            for pid in sorted(frozen):
                try:
                    os.kill(pid, signal.SIGCONT)
                except ProcessLookupError:
                    pass
