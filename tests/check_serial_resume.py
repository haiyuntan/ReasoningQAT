"""Local HTTP integration checks: resume, restart skip, ignored Range."""
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from threading import Thread

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from serial_download import download

DATA = b'0123456789abcdef' * 8192
requests = []


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        value = self.headers.get('Range')
        requests.append(value)
        start = int(value.split('=')[1].split('-')[0]) if value else 0
        self.send_response(200 if self.path == '/ignore' or not value else 206)
        self.send_header('Content-Length', str(len(DATA) - start))
        self.send_header('Content-Range', f'bytes {start}-{len(DATA)-1}/{len(DATA)}')
        self.end_headers()
        self.wfile.write(DATA[start:])

    def log_message(self, *args):
        pass


server = HTTPServer(('127.0.0.1', 0), Handler)
Thread(target=server.serve_forever, daemon=True).start()
try:
    with TemporaryDirectory() as folder:
        target = Path(folder) / 'weights'
        partial = Path(folder) / 'weights.part'
        partial.write_bytes(DATA[:12345])
        url = f'http://127.0.0.1:{server.server_port}'
        digest = hashlib.sha256(DATA).hexdigest()
        download(url + '/ok', target, len(DATA), digest, attempts=1)
        assert requests == ['bytes=12345-'] and target.read_bytes() == DATA
        download(url + '/ok', target, len(DATA), digest, attempts=1)
        assert len(requests) == 1, 'Completed verified file should be skipped'
        other = Path(folder) / 'other'
        other_partial = Path(folder) / 'other.part'
        other_partial.write_bytes(DATA[:100])
        try:
            download(url + '/ignore', other, len(DATA), digest, attempts=1)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Ignored Range should fail without appending')
        assert other_partial.read_bytes() == DATA[:100]
        print('PASS: byte-offset resume, completed-file skip, ignored-Range protection')
finally:
    server.shutdown()
