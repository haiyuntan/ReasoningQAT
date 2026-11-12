"""Use GPU idle during CPU Lean collection; fit only on preregistered discovery data."""
import os
from common import *
os.environ['ABCD_STATUS_FILE']='prefit_controller.json'
import pipeline

def main():
 while True:
  ready=W/'manifests/prepare.json'
  if ready.exists() and read(ready)['status']=='DONE':break
  controller=W/'manifests/controller.json'
  if controller.exists() and read(controller)['status'] in ['BLOCKED','STOPPED','STOPPED_RETRIEVAL_PRIORITY']:raise RuntimeError('Main preparation stopped')
  time.sleep(5)
 try:
  pipeline.run('prefit_P0','worker.py','--method','P0','--prepare-only',gpu=True,required=True)
  pipeline.write_controller({'pid':os.getpid(),'status':'DONE','stage':'A_fit_discovery_only'})
 except BaseException as error:
  pipeline.stop();pipeline.write_controller({'pid':os.getpid(),'status':'BLOCKED','error':repr(error)});raise
if __name__=='__main__':main()
