"""CPU-only paper-style DTR illustrations; never modify frozen observations."""
import csv
import hashlib
import json
import math
from pathlib import Path

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

W = Path(__file__).resolve().parents[2]
OUT = W / 'plots_paper_style'
PAPER = W / 'paper_sources/2602.13517v2.pdf'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def crop(doc, page, rect, name):
    result = fitz.open()
    result.new_page(width=rect.width, height=rect.height).show_pdf_page(
        fitz.Rect(0, 0, rect.width, rect.height), doc, page, clip=rect)
    result.save(OUT / (name + '.pdf'))
    result[0].get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72)).save(OUT / (name + '.png'))


def label(piece):
    # Spaces are kept as empty token columns. No token merging or re-tokenization.
    return piece.strip().replace('\n', r'\n').replace('\t', r'\t')


def heatmap(row, name, bits=True):
    original = np.asarray(row['jsd_matrix'], dtype=float)
    assert original.shape == (28, len(row['token_ids']))
    assert np.isfinite(original).all() and original.min() >= 0
    assert original.max() <= math.log(2) + 1e-5
    assert np.abs(original[-1]).max() < 1e-5
    matrix = original / math.log(2) if bits else original
    vmax = 1 if bits else math.log(2)
    fig = plt.figure(figsize=(12, 12 * 221 / 474))
    ax = fig.add_axes([.055, .035, .883, .845])
    im = ax.imshow(matrix, origin='lower', aspect='auto', cmap='YlOrBr',
                   vmin=0, vmax=vmax, interpolation='nearest',
                   extent=(-.5, matrix.shape[1]-.5, .5, 28.5))
    ax.set_yticks(list(range(1, 29, 2)))
    ax.set_ylabel('i-th Layer')
    ax.tick_params(axis='y', labelsize=9, length=2, width=.5)
    ax.set_xticks(range(matrix.shape[1]), [label(t) for t in row['token_strings']])
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', labelsize=9, length=2, width=.5, pad=2)
    for tick in ax.get_xticklabels():
        tick.set_rotation(20)
        tick.set_ha('left')
        tick.set_rotation_mode('anchor')
    for spine in ax.spines.values():
        spine.set_linewidth(.5)
    cax = fig.add_axes([.951, .035, .019, .845])
    cb = fig.colorbar(im, cax=cax, ticks=np.linspace(0, vmax, 6))
    cb.ax.tick_params(labelsize=9, length=2, width=.5)
    # Keep the unit inside the fixed canvas; a right-side label would be clipped.
    cb.ax.set_title('bits' if bits else 'nats', fontsize=8, pad=5)
    fig.savefig(OUT / (name + '.pdf'))
    fig.savefig(OUT / (name + '.png'), dpi=300)
    plt.close(fig)
    return matrix


def stack(top_path, bottom_path, name):
    top, bottom = fitz.open(top_path), fitz.open(bottom_path)
    width = 864
    heights = [width * doc[0].rect.height / doc[0].rect.width for doc in (top, bottom)]
    gap = 18
    result = fitz.open()
    page = result.new_page(width=width, height=sum(heights) + gap)
    page.show_pdf_page(fitz.Rect(0, 0, width, heights[0]), top, 0)
    page.show_pdf_page(fitz.Rect(0, heights[0]+gap, width, sum(heights)+gap), bottom, 0)
    result.save(OUT / (name + '.pdf'))
    page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72)).save(OUT / (name + '.png'))


def task_fig1(rows):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7))
    statistics = {}
    for ax, metric, title in zip(axes, ['token_length', 'dtr'],
                                 ['Token Count', 'Deep-Thinking Ratio']):
        correlations = []
        for model, color in [('P0', '#0072B2'), ('P4', '#D55E00')]:
            data = [r for r in rows if r['model'] == model and r['split'] == 'D-CONFIRM']
            x = np.array([float(r[metric]) for r in data])
            y = np.array([float(r['is_valid_transition']) for r in data])
            edges = np.unique(np.quantile(x, np.linspace(0, 1, 6)))
            groups = np.searchsorted(edges[1:-1], x, side='right')
            bins = [{'x': float(x[groups == i].mean()),
                     'validity': float(y[groups == i].mean()),
                     'n': int((groups == i).sum())}
                    for i in range(max(1, len(edges)-1)) if (groups == i).any()]
            bx = [b['x'] for b in bins]; by = [b['validity'] for b in bins]
            r = float(np.corrcoef(bx, by)[0, 1]) if len(bins)>1 and np.ptp(by)>0 else None
            if r is not None:
                correlations.append(r)
            ax.plot(bx, by, 'o-', color=color, label=f'{model}  r = {r:.3f}' if r is not None else model)
            statistics[model+'_'+metric] = {'bins': bins, 'pearson_of_bin_means': r,
                'n_theorems': len({r['theorem_id'] for r in data}), 'n_tactics': len(data)}
        ax.set_title(f'Avg Correlation r = {np.mean(correlations):.3f}')
        ax.set_xlabel(title); ax.set_ylabel('Lean tactic validity')
        ax.set_ylim(0, 1.03); ax.grid(alpha=.15); ax.legend(fontsize=9)
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(OUT / ('Fig1_REALProver_paper_style.'+ext), dpi=300)
    plt.close(fig)
    return statistics


def main():
    OUT.mkdir(exist_ok=True)
    inputs = [W/'metrics/per_layer_jsd.jsonl', W/'metrics/per_tactic_metrics.csv', PAPER]
    before = {str(p.relative_to(W)): digest(p) for p in inputs}
    rows = [json.loads(line) for line in inputs[0].open()]
    # Selection uses only split, readability, length and known Lean validity;
    # no JSD, DTR, entropy, convergence pattern or visual similarity is inspected.
    eligible = [r for r in rows if r['model']=='P0' and r['split']=='D-CONFIRM'
                and r['lean_status']=='VALID_TRANSITION' and 32<=len(r['token_ids'])<=40
                and all('\ufffd' not in t for t in r['token_strings'])]
    assert eligible, 'No readable, valid long-tactic candidate in frozen confirmation data'
    selected = min(eligible, key=lambda r: r['candidate_id'])
    paired = next(r for r in rows if r['candidate_id']==selected['candidate_id'] and r['model']=='P4')
    assert selected['token_ids']==paired['token_ids']
    doc = fitz.open(PAPER)
    crop(doc, 0, fitz.Rect(98, 440, 499, 666), 'DTR_source_Fig1')
    crop(doc, 2, fitz.Rect(60, 83, 534, 304), 'DTR_source_Fig2')
    for row in [selected, paired]:
        model = row['model']
        values = heatmap(row, f'Fig2_REALProver_{model}_paper_style')
        np.testing.assert_allclose(values*math.log(2), np.array(row['jsd_matrix']), rtol=1e-14, atol=1e-14)
        heatmap(row, f'Fig2_REALProver_{model}_nats', bits=False)
        stack(OUT/'DTR_source_Fig2.pdf', OUT/f'Fig2_REALProver_{model}_paper_style.pdf',
              f'Fig2_DTR_top_REALProver_{model}_bottom')
    with inputs[1].open() as stream:
        statistics = task_fig1(list(csv.DictReader(stream)))
    after = {str(p.relative_to(W)): digest(p) for p in inputs}
    assert before == after
    manifest = {'status': 'PASS', 'source_url': 'https://arxiv.org/pdf/2602.13517v2',
        'paper_version': 'v2, 2026-07-06', 'source_pdf_sha256': digest(PAPER),
        'source_crops_pdf_points': {'Fig1': [98,440,499,666], 'Fig2': [60,83,534,304]},
        'candidate_id': selected['candidate_id'], 'theorem_id': selected['theorem_id'],
        'selection_rule': 'P0 confirmation, VALID_TRANSITION,32..40 native tokens, no isolated replacement characters; smallest candidate ID. No JSD/DTR-based filtering.',
        'eligible_candidates': len(eligible), 'generated_by': 'P4',
        'P0_measurement': 'same-token replay; not independently generated P0',
        'shown_interval_zero_based_half_open': [0,len(selected['token_ids'])],
        'tactic_text': selected['tactic_text'], 'token_ids': selected['token_ids'],
        'token_strings': selected['token_strings'], 'layers': selected['layer_ids'],
        'plot_units': 'bits = stored nats / ln(2); no rescaling by observed min/max',
        'original_figure_units': 'Colorbar 0..1; author log-base not established by the crop alone',
        'nats_alternatives_provided': True, 'measurement': 'normalized logit lens; full-vocabulary JSD',
        'primary_statistics_unchanged': True, 'inputs_unchanged': before==after,
        'input_sha256': before, 'Fig1_statistics': statistics,
        'script_sha256': digest(Path(__file__)),
        'heatmap_cases': {r['model']: {'dtr_nats_g05': r['dtr'],
            'matrix_nats': r['jsd_matrix']} for r in [selected,paired]}}
    manifest['output_sha256'] = {p.name:digest(p) for p in sorted(OUT.iterdir()) if p.suffix in ['.pdf','.png']}
    (OUT/'source_and_validation.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))
    (OUT/'README.zh.md').write_text('''# DTR 论文风格图\n\n主对照图：Fig2_DTR_top_REALProver_P0_bottom.png（另有 PDF）。上半幅是原论文 Fig2 的原始 PDF 裁图，下半幅是 REALProver P0 对同一 P4 生成 tactic 的真实逐层测量；不是独立 P0 自然生成。另提供 P4 自然生成测量版本。\n\nDTR_source_Fig1.png/pdf 是原论文 Fig1 原图；DTR_source_Fig2.png/pdf 是原论文 Fig2 原图。来源：https://arxiv.org/pdf/2602.13517v2（2026-07-06），仅裁剪图像区域，未重绘或修改作者数值。\n\n下方热图对齐原图的 YlOrBr 配色、顶部原生 token 标签、浅层在下/深层在上、竖直色条及单幅布局。所有 28 层、连续完整 tactic 的全部 token 均保留，空格 token 保留空列，不插值、不重排、不合并。为避免不完整 UTF-8 BPE 字节误标，选择无替换字符的样本。具体确定性筛选规则、全部 token、原始矩阵和输入哈希见 source_and_validation.json；筛选不依据 JSD、DTR 或热图相似度。\n\n彩条 0–1 的版本明确使用 JSD bits，即原始 nats 除以 ln(2)，不是按本图最大最小值拉伸。原论文图色条虽为 0–1，但仅凭 PDF 不能确定作者使用的对数底数，因此不能声称两图阈值/单位已完全复现。另提供 Fig2_REALProver_P0_nats 与 P4_nats，保留 0–ln(2) 的原始单位。原实验 g=0.5 nats 与 DTR 统计不变；若在 bits 上解释同一阈值，应为 0.5/ln(2)，不是 0.5 bits。\n\nFig1_REALProver_paper_style 使用确认集 20 个证明状态、160 条 tactic，按指标五分位分箱（相同值不拆分），计算分箱均值之间的 Pearson r，供对齐论文 Fig1 的呈现方式。纵轴明确为 tactic 有效性，不是完整证明 pass@1。P0/P4 使用相同候选及 Lean 标签，所以长度曲线重叠；不能当作独立两模型准确率比较，亦不能通过调整样本获得预期正相关。原先统计和图不覆盖。\n\n复现：使用同时有 PyMuPDF、NumPy、Matplotlib 的 Python 运行 src/figures/build_paper_comparison.py。需要 paper_sources/2602.13517v2.pdf 与已有冻结 metrics。全流程 CPU，无新增模型推理，不干扰正式评估。\n''')
    print(json.dumps({'status':'PASS','candidate_id':selected['candidate_id'],
        'tokens':len(selected['token_ids']),'figures':len(manifest['output_sha256']),
        'inputs_unchanged':True}))


if __name__ == '__main__':
    main()
