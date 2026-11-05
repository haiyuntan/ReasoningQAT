# 精确复现需要确认的最小信息

1. Fig.2(b)原始绘图脚本，或原始二维坐标、domain标签和对应输入ID。
2. Qwen3-1.7B的具体revision；FineWeb-Edu/OpenThoughts3版本、split、shuffle seed及样本ID；每域样本/点数。
3. 点是token、序列还是通道？问题/回答是否都输入？原始token位置如何抽样？是否包含模板及特殊token？
4. 第14块是one-based还是zero-based？提取块输入、块输出、归一化输入、attention/MLP内部哪个张量？是否池化？
5. 标准化/PCA、距离度量、perplexity、learning_rate、init、seed、early_exaggeration、迭代数与库版本。
6. 色条0..5的真实含义与各域ID；marker size、alpha、绘制顺序、是否作显示旋转/缩放及坐标轴限制。

以上仅为待确认清单，未自动向作者发送消息或发issue。
