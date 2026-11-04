import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

def test_latency_tolerates_transient_and_resets():
    from resource_guard import LatencyWindow
    window=LatencyWindow(2.69)
    assert window.soft==10
    for seconds in [3.45,11,12,2,11,12,3]:window.observe(seconds)
    assert window.consecutive==0

def test_latency_stops_persistent_or_severe_regression():
    import pytest
    from resource_guard import LatencyWindow
    window=LatencyWindow(2)
    window.observe(11);window.observe(12)
    with pytest.raises(RuntimeError,match='consecutive'):window.observe(13)
    with pytest.raises(RuntimeError,match='hard'):LatencyWindow(2).observe(20)

def test_retrieval_failure_signals_only_created_child(monkeypatch,tmp_path):
    import resource_guard as g
    for name in ['tmp','logs','manifests']:(tmp_path/name).mkdir()
    class Child:
        pid=987654321
        returncode=None
        def poll(self):return self.returncode
        def wait(self,timeout=None):self.returncode=-15;return self.returncode
    child=Child();signals=[];records=[]
    monkeypatch.setattr(g,'OBS',tmp_path)
    monkeypatch.setattr(g,'tail_events',lambda:{})
    monkeypatch.setattr(g,'read',lambda p:{'status':'RUNNING','fingerprint':'test'})
    monkeypatch.setattr(g,'atomic',lambda p,v:records.append(v))
    monkeypatch.setattr(g.subprocess,'Popen',lambda *a,**k:child)
    monkeypatch.setattr(g.os,'setpriority',lambda *a:None)
    monkeypatch.setattr(g.os,'killpg',lambda pid,sig:signals.append((pid,sig)))
    monkeypatch.setattr(g.time,'sleep',lambda n:None)
    def failure(*a,**k):raise TimeoutError('retrieval degraded')
    monkeypatch.setattr(g,'request',failure)
    result=g.supervise(['observation-only'],{'gpu':{'uuid':'test-uuid'},'fingerprint':'test','probe_p95':1},'test')
    assert result['status']=='STOPPED_RETRIEVAL_PRIORITY'
    assert len(signals)==1 and signals[0][0]==child.pid

def test_offset_categories_multitoken_identifier():
    import pytest
    pytest.importorskip('pyarrow')
    from select_observation_data import classify
    text='rw [Nat.add_comm]'
    labels=classify(text,[(0,2),(4,7),(7,11),(11,16)])
    assert labels[0]=='tactic_keyword'
    assert labels[1:]==['theorem_reference']*3
