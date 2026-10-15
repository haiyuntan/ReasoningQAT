from flask import Flask, request, jsonify
from worker import PremiseSelector
from util import RequestUtil
import os

app = Flask(__name__)

premise_selector = PremiseSelector()
premise_selector.init_model()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ready', 'premises': premise_selector.index.ntotal,
                    'adapter_loaded': bool(premise_selector.model.peft_config),
                    'asset_manifest_sha256': premise_selector.asset_manifest_sha256,
                    'source_revision': '3e5987dc1009a7addc935bcd7b4ae777738a0848'})

@app.route('/retrieve_premises', methods=['POST'])
def to_formal_statement():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify(RequestUtil.gen_fail_data('Missing param \'query\'')), 400
    num = data.get('num', 5)
    related_theorems = premise_selector.retrieve(data['query'], num=num)
    return jsonify(RequestUtil.gen_success_data(related_theorems)), 200

# Start http server
if __name__ == '__main__':
    port = int(os.environ.get('LEANSEARCH_PORT', '8080'))
    print(f'start server on {port}')
    app.run(host='127.0.0.1', port=port, debug=False, use_reloader=False, threaded=False)
