"""Compose existing vector figures without changing any measured values."""
import hashlib
import json
import argparse
from pathlib import Path
import fitz

W = Path(__file__).resolve().parents[2]
OUT = W / 'plots_paper_style'
PANELS = [
    ('DTR_source_Fig1.pdf', '(a) DTR paper: Fig. 1'),
    ('DTR_source_Fig2.pdf', '(b) DTR paper: Fig. 2'),
    ('Fig2_REALProver_P0_paper_style.pdf', '(c) REALProver P0 (BF16)'),
    ('Fig2_REALProver_P4_paper_style.pdf', '(d) REALProver P4 (W2G128)'),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quantized-model', choices=['P4', 'P1'], default='P4')
    model = parser.parse_args().quantized_model
    panels = list(PANELS)
    replay = None
    if model == 'P1':
        from build_paper_comparison import heatmap
        replay = json.loads((W/'metrics/P1_figure_replay.json').read_text())
        selection = json.loads((OUT/'source_and_validation.json').read_text())
        assert replay['status']=='DONE' and replay['model']=='P1'
        assert replay['candidate_id']==selection['candidate_id'] and replay['token_ids']==selection['token_ids']
        heatmap(replay, 'Fig2_REALProver_P1_paper_style')
        panels[3] = ('Fig2_REALProver_P1_paper_style.pdf', '(d) REALProver P1 (GPTQ W2G128)')
    width, height, margin, gap, header = 660, 376, 18, 18, 25
    doc = fitz.open()
    page = doc.new_page(width=2*width+2*margin+gap,
                        height=2*(height+header)+2*margin+gap)
    inputs = {}
    for i, (name, title) in enumerate(panels):
        source = OUT/name
        inputs[name] = sha(source)
        col, row = i % 2, i // 2
        x = margin+col*(width+gap)
        y = margin+row*(height+header+gap)
        page.insert_text((x+4, y+15), title, fontsize=13, fontname='helv')
        panel = fitz.open(source)
        page.show_pdf_page(fitz.Rect(x, y+header, x+width, y+header+height),
                           panel, 0, keep_proportion=True)
    stem = OUT/f'DTR_four_panel_source_Fig1_Fig2_P0_{model}'
    doc.save(stem.with_suffix('.pdf'), garbage=4, deflate=True)
    page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72)).save(stem.with_suffix('.png'))
    assert all(sha(OUT/name)==value for name,value in inputs.items())
    previous = json.loads((OUT/'source_and_validation.json').read_text())
    manifest = dict(status='PASS', layout=f'paper Fig1 | paper Fig2 / P0 BF16 | {model} W2G128',
        selected_quantized_model=model,
        reason='Paired measurements of identical prompt and generated token sequence.',
        candidate_id=previous['candidate_id'], input_sha256=inputs,
        same_token_pairing=True, P0_measurement='same-token replay of P4-generated tactic',
        generation_origin='P4', quantized_measurement='natural P4 generation' if model=='P4' else 'same-token P1 replay', display_units='bits; frozen primary metrics remain nats',
        source_paper='https://arxiv.org/pdf/2602.13517v2',
        changes='Composition and minimal panel identifiers only; vector panels kept in original proportions.',
        output_sha256={ext:sha(stem.with_suffix('.'+ext)) for ext in ['png','pdf']},
        script_sha256=sha(Path(__file__)))
    if replay is not None:
        manifest['P1_measurement_sha256'] = sha(W/'metrics/P1_figure_replay.json')
        manifest['P1_checkpoint_manifest_sha256'] = sha(W/'manifests/models_P1_figure.json')
    stem.with_suffix('.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps({'status':'PASS','outputs':[str(stem.with_suffix('.'+x)) for x in ['png','pdf']]}))


if __name__ == '__main__':
    main()
