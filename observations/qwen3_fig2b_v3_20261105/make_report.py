"""Build comparisons and a complete, non-selected gallery."""
import json,hashlib
from explore import OUT,ROOT,save
import fitz
import matplotlib.pyplot as plt

def main():
    style=OUT/'paper_style';evidence=json.loads((OUT/'reference_audit.json').read_text())
    rect=evidence['embedded_image_pdf_rect'];ratio=488/(rect[2]-rect[0])
    doc=fitz.open(ROOT/'papers/reasoningqat_iclr2026.pdf')
    clip=fitz.Rect(rect[0],rect[1]-50/ratio,rect[2],rect[3])
    doc[2].get_pixmap(matrix=fitz.Matrix(ratio*2,ratio*2),clip=clip).save(str(OUT/'paper_reference_with_heading.png'))
    names=['pca50_initrandom_lr200_perp30','pcaNone_initrandom_lr200_perp30']
    paths=[OUT/'paper_reference_with_heading.png']+[style/(n+'_annotated.png') for n in names]
    fig=plt.figure(figsize=(15,5.2))
    for i,(path,title) in enumerate(zip(paths,['Original Fig.2(b)','Same 2048 points: PCA50','Same 2048 points: no PCA'])):
        ax=fig.add_axes([.01+i*.33,.12,.32,.76]);ax.imshow(plt.imread(path));ax.axis('off')
        fig.text(.17+i*.33,.94,title,ha='center',fontsize=12)
    fig.text(.5,.025,'Source: Okoshi et al., ICLR2026, Fig.2(b). Reconstructed panels use fixed seed42, random init, LR200, 1000 iterations.\nDisplay-aligned by global rotation/reflection and uniform scaling; native coordinates and every control are retained.',ha='center',fontsize=9)
    fig.savefig(OUT/'paper_vs_pca_vs_no_pca.png',dpi=180);plt.close(fig)
    fig=plt.figure(figsize=(10,5.2))
    for i,(name,title) in enumerate([('nopca_legacy_native_coordinates','Native coordinates'),('pcaNone_initrandom_lr200_perp30','Display-aligned coordinates')]):
        ax=fig.add_axes([.01+i*.5,.08,.48,.8]);ax.imshow(plt.imread(style/(name+'_annotated.png')));ax.axis('off');fig.text(.25+i*.5,.94,title,ha='center',fontsize=12)
    fig.savefig(OUT/'native_vs_display.png',dpi=180);plt.close(fig)
    results=json.loads((OUT/'results.json').read_text());robust=json.loads((OUT/'robustness_results.json').read_text());audit=json.loads((OUT/'pca_neighborhood_audit.json').read_text())
    text='''# Fig.2(b)：PCA诊断与原图版式对齐

本轮完成：10组固定参数对照、3组无PCA种子/迭代检查；另保留v2主图及原始坐标、未旋转对照。**无PCA的版本在二维外观上更接近原图，但仍不是作者精确配置的确认复现。**

## 原图、相同点的PCA与无PCA比较

![三图比较](paper_vs_pca_vs_no_pca.png)

中、右两图使用完全相同的2048个token向量、seed42、random初始化、学习率200、1000次迭代，只改变是否先PCA50。展示方向与比例另作全局对齐，详细说明见下文。图形相似不能证明作者就是采用这组参数。

## 重点查看：不先做PCA的历史默认参数对照

![无PCA对照](paper_style/pcaNone_initrandom_lr200_perp30_annotated.png)

[PNG纯图](paper_style/pcaNone_initrandom_lr200_perp30.png) · [PDF](paper_style/pcaNone_initrandom_lr200_perp30.pdf) · [SVG](paper_style/pcaNone_initrandom_lr200_perp30.svg)

该参数组在探索前已列入固定对照，没有通过挑种子或移动某个域来匹配原图。它是可检查的候选重建，不替代此前预先指定的v2主配置及全部历史结果。

## 发现的实质问题

'''
    text+=f"PCA50仅保留当前点集方差的 **{audit['pca50_explained_variance_fraction']:.1%}**，与原始空间15近邻的平均重合为 **{audit['mean_original_vs_pca50_15nn_overlap']:.1%}**。因此它不是无影响的绘图步骤，可能明显改变t-SNE所拟合的邻域。本轮优先检查无PCA，是依据这个诊断，不是只依据外观。\n\n"
    text+=f"同一2048点原始高维合并推理/预训练离散度为{audit['original_high_dimensional_reasoning_to_pretraining_dispersion']:.3f}，PCA50后为{audit['pca50_reasoning_to_pretraining_dispersion']:.3f}，均小于1。二维图的排序不能外推为高维空间中的同样排序。\n\n"
    text+='''## 已对齐的版式与其证据

- 顶部`Dataset Domain`及`Pre-training / Code / Math / Science`：直接读取原PDF的文字原点、字号、RGB、下划线矩形；本机从PDF嵌入子集字体恢复Unicode映射，以原YuGothic-Bold字形重绘。字体二进制不上传。
- 横轴−40..40、步长20；纵轴−40..40、步长10；主轴与色条的长宽及间距按嵌入图488×357像素测量。
- 右侧连续Viridis色条0..5；由原JPEG内颜色拟合出的透明度约0.201，RGB拟合RMSE低于1/255量级（数值为约0.99个8bit灰阶）。
- 为复刻可见配色，推断Code=0、Math=1、Science=2、Pre-training=5。**这是绘图编码假设，不是论文确认的编号定义，更不是熵、概率或激活强度。** 3/4只在我们的映射中未使用，不能据此推断作者没有其他编号。
- 散点透明度0.2由色条联动推断，尺寸36为默认尺寸假设；作者原始散点参数仍未知。

## 坐标和位置的对齐方式

为了把预训练域展示在下方、Code展示在Science左侧，展示版只施加**一套作用于所有点的正交旋转/镜像**，再居中并作**同一个等比例缩放**以完整放入−40..40。Math等其他域不单独定位。成对距离按同一比例保持，且没有裁掉数据点。它是有类别锚点的版式处理，不是原始坐标或独立复现的证据。

![原始和展示坐标](native_vs_display.png)

每幅图对应的NPZ同时保存`native_xy`、`display_xy`和标签；JSON记录矩阵、平移、缩放、随机成对距离校验和裁切计数。需要做科学距离统计时使用原始特征/坐标，不能把显示坐标当作原始t-SNE数值。各图独立归一化显示，不应跨图比较坐标绝对大小。

## 全部10组固定参数对照

均为相同2048点，seed42，1000次迭代；每幅有PNG、PDF、SVG，且不按外观相似度隐藏其他结果。

| PCA | 初始化 | LR | perplexity | KL | 二维推理/预训练离散度 | 图片 |
|---|---|---|---:|---:|---:|---|
'''
    for r in results:text+=f"| {r['pca']} | {r['init']} | {r['lr']} | {r['perplexity']} | {r['kl_divergence']:.3f} | {r['two_dimensional_reasoning_to_pretraining_dispersion']:.3f} | [图片](paper_style/{r['tag']}_annotated.png) |\n"
    text+='\n不同PCA输入对应不同邻域图，其KL不能作为共同准确性评分来选胜者。\n\n## 无PCA稳定性检查\n\n| seed | 迭代 | KL | 二维推理/预训练离散度 | 图片 |\n|---:|---:|---:|---:|---|\n'
    for r in robust:text+=f"| {r['seed']} | {r['iterations']} | {r['kl_divergence']:.3f} | {r['two_dimensional_reasoning_to_pretraining_dispersion']:.3f} | [图片](paper_style/{r['tag']}_annotated.png) |\n"
    text+='''
## 基础模型与官方源码核对

本机权重两个分片、config、generation_config、tokenizer配置及tokenizer内容均与官方匹配。tokenizer.json为LFS内容，应核对LFS SHA256，不能把Git指针blob哈希与文件内容哈希混淆。[模型审计](base_model_audit.json)。

官方当前主分支17个Python文件无t-SNE关键字；两个可见历史版本文件路径也未发现绘图脚本。[内容审计](official_keyword_audit.json) · [历史审计](official_history_audit.json)。OpenReview附件API返回403，未声称已穷尽不可访问附件。

## 为什么仍不能保证逐点精确复现

仍缺作者的样本ID/数据版本、选取的token位置、块内激活张量定义、是否池化/标准化，以及t-SNE库版本和全部参数。仅凭原图不能反推出唯一解。色条编号、散点尺寸和透明度也尚非作者代码确认。

[需要作者确认的最小信息](AUTHOR_QUESTIONS.md) · [版式验证](exact_style_validation.json) · [原图审计](reference_audit.json) · [原始t-SNE坐标及变换](coordinates/) · [PCA邻域审计](pca_neighborhood_audit.json)。

本轮所有计算仅CPU，复用已有激活；检索GPU、正式实验代码/环境/权重/结果均未改动。

来源：[论文Fig.2(b)](https://proceedings.iclr.cc/paper_files/paper/2026/file/209423f076b6479ab3a4f45886e30306-Paper-Conference.pdf)、[官方仓库](https://github.com/yasu0001/ReasoningQAT)、[sklearn t-SNE参数文档](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)。
'''
    (OUT/'README.md').write_text(text)
    (OUT/'AUTHOR_QUESTIONS.md').write_text('''# 精确复现需要确认的最小信息

1. Fig.2(b)原始绘图脚本，或原始二维坐标、domain标签和对应输入ID。
2. Qwen3-1.7B的具体revision；FineWeb-Edu/OpenThoughts3版本、split、shuffle seed及样本ID；每域样本/点数。
3. 点是token、序列还是通道？问题/回答是否都输入？原始token位置如何抽样？是否包含模板及特殊token？
4. 第14块是one-based还是zero-based？提取块输入、块输出、归一化输入、attention/MLP内部哪个张量？是否池化？
5. 标准化/PCA、距离度量、perplexity、learning_rate、init、seed、early_exaggeration、迭代数与库版本。
6. 色条0..5的真实含义与各域ID；marker size、alpha、绘制顺序、是否作显示旋转/缩放及坐标轴限制。

以上仅为待确认清单，未自动向作者发送消息或发issue。
''')
    groups=[{'domain':r['domain'],'sha256':r['sha256'],'source':r['source'],'shard':str(r['path']).split('/')[-1]} for r in [json.loads(s) for s in (ROOT/'fig2b_qwen3_v2_20261105/inputs.jsonl').read_text().splitlines()]]
    save('sample_groups.json',groups)
    save('input_identity.json',{'v2_inputs_sha256':hashlib.sha256((ROOT/'fig2b_qwen3_v2_20261105/inputs.jsonl').read_bytes()).hexdigest(),
        'groups':512,'tokens_per_group':4,'group_order':'same as sample_groups.json','token_selection':'first4 of16 previously uniformly sampled positions; all four domains128 groups each'})
if __name__=='__main__':main()
