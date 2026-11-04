"""Replot published coordinates/statistics on CPU; no weights or GPU required."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parent/'all_layers_corrected'
    colors={'General':'tab:blue','Math':'tab:orange','Code':'tab:green','Lean':'tab:red'}
    for layer in [3,14]:
        data=pd.read_csv(root/'metrics'/f'layer{layer}_tsne_coordinates.csv')
        assert len(data)==1024 and data.sample_id.nunique()==1024
        fig,ax=plt.subplots(figsize=(8,6))
        for domain,color in colors.items():
            part=data[data.domain==domain]
            ax.scatter(part.x_seed2,part.y_seed2,s=12,alpha=.7,label=domain,color=color)
        ax.set_title(f'Figure 2(b)-style: REAL-Prover BF16, Block {layer}')
        ax.set_xticks([]);ax.set_yticks([]);ax.legend(frameon=False)
        fig.text(.5,.01,'256 groups/domain; published seed-2 coordinates; no t-SNE refitting\nExploratory Block 3 selection; Block 14 reference; Lean in training distribution',ha='center',fontsize=8)
        fig.tight_layout(rect=[0,.07,1,1])
        fig.savefig(args.output/f'realprover_bf16_layer{layer}_tsne.png',dpi=220);plt.close(fig)
    data=pd.read_csv(root/'metrics/all_layer_domain_statistics.csv')
    assert data.layer.tolist()==list(range(1,29))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for tag in ['chat','raw','last']:axes[0].plot(data.layer,data[tag+'_silhouette'],label=tag)
    for tag in ['chat','raw']:axes[1].plot(data.layer,data[tag+'_probe_cv_mean'],label=tag)
    for ax in axes:
        ax.axvline(3,color='gray',linestyle='--');ax.set_xlabel('Decoder block (one based)');ax.legend()
    axes[0].set_ylabel('High-dimensional cosine silhouette');axes[1].set_ylabel('5-fold domain probe accuracy')
    fig.tight_layout();fig.savefig(args.output/'all_layer_domain_metrics.png',dpi=200);plt.close(fig)
    print('Wrote 3 plots to',args.output)

if __name__=='__main__':main()
