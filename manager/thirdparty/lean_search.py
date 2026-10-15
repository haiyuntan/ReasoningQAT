import time
import os
import json
import conf.config
from util import StringUtil, HttpUtil

REQUEST_URL = conf.config.API_CONFIG['lean_search']


class LeanSearch:

    @staticmethod
    def get_related_theorem(query: str):
        data = LeanSearch.get_param(query)
        try:
            result = HttpUtil.post(url=REQUEST_URL, json=data)
            premises = result['data'][0]
            if not isinstance(premises, list) or not premises:
                raise RuntimeError('Retrieval returned invalid or empty premises')
        except Exception:
            LeanSearch._telemetry(False)
            raise
        LeanSearch._telemetry(True)
        return premises

    @staticmethod
    def _telemetry(success):
        path = os.environ.get('REALPROVER_RETRIEVAL_TELEMETRY')
        if path:
            # One atomic append per request across evaluator child processes.
            fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                os.write(fd, (json.dumps({'success': success, 'time': time.time()}) + '\n').encode())
            finally:
                os.close(fd)

    @staticmethod
    def get_param(query: str, num: int=conf.config.NUM_QUERYS):
        params = {
            'timestamp': time.time(),
            'query': query,
            'num': num
        }
        params['sign'] = StringUtil.gen_sign(params)
        return params

    @staticmethod
    def get_related_theorem_batch(queries: list[str]):
        data = [LeanSearch.get_param(q) for q in queries]
        result = HttpUtil.post(url=REQUEST_URL, json=data)
        return result['data'] # type: ignore

if __name__ == '__main__':
    # TODO: write tests here!
    pass
