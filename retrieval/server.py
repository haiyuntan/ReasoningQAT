from flask import Flask, request, jsonify
from worker import PremiseSelector
from util import RequestUtil
import os
import threading
import json
import time

app = Flask(__name__)

premise_selector = PremiseSelector()
premise_selector.init_model()
retrieval_lock = threading.Lock()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ready', 'premises': premise_selector.index.ntotal,
                    'adapter_loaded': bool(premise_selector.model.peft_config),
                    'asset_manifest_sha256': premise_selector.asset_manifest_sha256,
                    'search_backend': premise_selector.runtime['search_backend'],
                    'runtime_sha256': premise_selector.runtime_sha256,
                    'cache_entries': len(premise_selector.cache.values),
                    'source_revision': '3e5987dc1009a7addc935bcd7b4ae777738a0848'})

@app.route('/retrieve_premises', methods=['POST'])
def to_formal_statement():
    started = time.perf_counter()
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify(RequestUtil.gen_fail_data('Missing param \'query\'')), 400
    num = data.get('num', 5)
    # Keep model/FAISS requests serialized while allowing cheap health requests
    # to respond independently of a long embedding/search request.
    with retrieval_lock:
        acquired = time.perf_counter()
        related_theorems = premise_selector.retrieve(data['query'], num=num)
        metrics = dict(premise_selector.last_metrics)
    metrics.update(queue_seconds=acquired-started, total_seconds=time.perf_counter()-started,
                   timestamp=time.time(), success=True)
    target = os.environ.get('REALPROVER_RETRIEVAL_PROFILE')
    if target:
        fd = None
        try:
            fd = os.open(target, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            os.write(fd, (json.dumps(metrics)+'\n').encode())
        except OSError as error:
            app.logger.warning('Retrieval timing log unavailable: %s', error)
        finally:
            if fd is not None:
                os.close(fd)
    response = jsonify(RequestUtil.gen_success_data(related_theorems))
    for key in ('queue_seconds', 'total_seconds', 'encoding_seconds', 'search_seconds', 'cache_hit'):
        response.headers['X-RealProver-'+key] = str(metrics[key])
    return response, 200

# Start http server
if __name__ == '__main__':
    port = int(os.environ.get('LEANSEARCH_PORT', '8080'))
    print(f'start server on {port}')
    app.run(host='127.0.0.1', port=port, debug=False, use_reloader=False, threaded=True)
