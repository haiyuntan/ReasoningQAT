"""Export real confirmation estimates and intervals; no synthetic data."""
from common import *
def main():
 import numpy as np
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 s=read(W/'metrics/confirmation_summary.json');methods=[m for m in P['models'][1:] if s.get(m,{}).get('status')=='CONFIRMATION_SCORES_DONE']
 fig,axes=plt.subplots(2,2,figsize=(11,8));colors=['#4477AA','#EE6677','#228833','#AA3377']
 panels=[('A','Paired correlation difference (layer28)'),('B','Alpha-renaming excess sensitivity'),('C','Removal minus same-length control')]
 for index,(key,title) in enumerate(panels):
  ax=axes.flat[index]
  for j,m in enumerate(methods):
   if key=='A':r=s[m]['A']['28'];mean=r.get('paired_difference');ci=r.get('simultaneous_ci')
   elif key=='B':r=s[m]['B']['primary_length_matched_family_clustered'];mean=r.get('mean');ci=r.get('simultaneous_ci')
   else:r=s[m]['C']['remove_minus_same_length'];mean=r.get('mean');ci=r.get('simultaneous_ci')
   if mean is None or ci is None:continue
   ax.vlines(j,ci[0],ci[1],color=colors[j]);ax.hlines([ci[0],ci[1]],j-.08,j+.08,color=colors[j]);ax.plot(j,mean,'o',color=colors[j])
  ax.axhline(0,color='gray',linewidth=.8);ax.set_xticks(range(len(methods)),methods);ax.set_title(title)
 ax=axes[1,1]
 for j,m in enumerate(methods):
  if s[m]['D']['status']!='EXACT_REPLAY_PASS':continue
  values=[s[m]['D']['per_layer'][str(l)]['excess_over_orthogonal'].get('mean',float('nan')) for l in range(1,29)]
  ax.plot(range(1,29),values,label=m,color=colors[j])
 ax.axhline(0,color='gray',linewidth=.8);ax.set_title('D: excess over orthogonal baseline');ax.set_xlabel('Block');ax.legend()
 fig.tight_layout();fig.savefig(W/'plots/ABCD_confirmation.png',dpi=200);fig.savefig(W/'plots/ABCD_confirmation.pdf');plt.close(fig)
 status('plots','DONE',methods=methods)
if __name__=='__main__':main()
