"""Persist bounded Lean timeouts as unsuccessful, budget-consuming attempts."""


def timeout_result(statement, generator, error):
    return {
        'formal_statement': statement,
        'collect_results': [{'declaration': None, 'success': False,
                             'calls': list(generator.calls), 'nodes': [],
                             'stop_cause': {'lean_response_timeout': True}}],
        'pipeline_verifier': {'executed': False, 'success': False},
        'pipeline_attempt': {'status': 'timeout', 'consumes_attempt': True,
                             'operation': error.operation, 'timeout_seconds': error.seconds,
                             'error': str(error)},
    }
