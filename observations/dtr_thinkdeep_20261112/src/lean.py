"""Independent bounded Lean sessions. Transport failures never become invalid."""
import os,sys,re
from common import *
os.environ.update(REALPROVER_LEAN_WORKSPACE=str(W/'runtime/lean'),REALPROVER_INTERACTIVE_ROOT='/tmp/realprover_lean_v10/interactive',REALPROVER_LEAN_BIN='/tmp/realprover_lean_v10/toolchain/bin')
os.environ['PATH']='/tmp/realprover_lean_v10/toolchain/bin:'+os.environ['PATH']
sys.path.insert(0,str(W/'upstream/realprover'))
from manager.thirdparty.interactive import Interactive,InteractiveResponseTimeout,InteractiveSessionError
from manager.struct.structs import state_repr
class Session:
    def __init__(self,case):
        self.path=W/'runtime/lean'/('Case_'+digest(case['formal_statement'])[:16]+'.lean');self.path.write_text(case['formal_statement']);self.s=None
        try:
            self.s=Interactive(W/'runtime/lean',Path('Header.lean'));self.s.response_timeout=30
            self.s.open_file(self.path,[None]);assert self.s.get_next_problem() is not None
            self.state=state_repr(self.s.get_state(0))
            if case.get('state'):assert self.state==case['state'],'ROOT_STATE_MISMATCH'
        except BaseException:self.close();raise
    def verdict(self,tactic):
        if re.search(r'\b(sorry|admit)\b',tactic):return dict(lean_status='INVALID_TACTIC',error_type='FORBIDDEN_PROOF_ESCAPE',goals_remaining=None,kernel_verified=False)
        try:
            self.s.response_timeout=20;sid=self.s.run_tactic(0,tactic,heartbeats=200000000);goals=self.s.get_state(sid)
            return dict(lean_status='VALID_TRANSITION',goals_remaining=len(goals),goals_closed=not goals,kernel_verified=False,successor_hash=digest(state_repr(goals)))
        except InteractiveResponseTimeout as e:return dict(lean_status='TIMEOUT',error=str(e),goals_remaining=None,kernel_verified=False)
        except InteractiveSessionError as e:return dict(lean_status='UNKNOWN_TRANSPORT',error=str(e),goals_remaining=None,kernel_verified=False)
        except RuntimeError as e:
            text=str(e);low=text.lower();kind='OTHER_LEAN_ERROR'
            for label,term in [('TIMEOUT','maximum number of heartbeats'),('SYNTAX','unexpected token'),('UNKNOWN_CONSTANT','unknown constant'),('TYPE_MISMATCH','type mismatch')]:
                if term in low:kind=label;break
            return dict(lean_status='TIMEOUT' if kind=='TIMEOUT' else 'INVALID_TACTIC',error_type=kind,error=text[:1200],goals_remaining=None,kernel_verified=False)
    def close(self):
        if self.s is not None:self.s.close();self.s=None
        self.path.unlink(missing_ok=True)
_session=None
_session_key=None
_verdict_cache={}
def close_reused():
    global _session
    if _session is not None:_session.close();_session=None
import atexit
atexit.register(close_reused)
def verify(case,tactic):
    global _session,_session_key,_verdict_cache
    key=digest(case['formal_statement'])
    if key!=_session_key:
        close_reused();_session_key=key;_verdict_cache={}
    if tactic in _verdict_cache:return dict(_verdict_cache[tactic])
    try:
        if _session is None:_session=Session(case)
        result=_session.verdict(tactic)
        if result['lean_status'] in ['TIMEOUT','UNKNOWN_TRANSPORT']:close_reused()
        else:_verdict_cache[tactic]=dict(result)
        return result
    except (InteractiveResponseTimeout,TimeoutError) as e:return dict(lean_status='TIMEOUT',error=str(e),kernel_verified=False,goals_remaining=None)
    except InteractiveSessionError as e:return dict(lean_status='UNKNOWN_TRANSPORT',error=str(e),kernel_verified=False,goals_remaining=None)
    finally:pass
