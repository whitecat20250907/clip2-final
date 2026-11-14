# CLIP2 T019 final frozen system

最终冻结算法的私有归档。保留原科学代码、训练得到的模型/适配器、精确配置、最终预测和依赖清单。未重新训练，未改变融合配方，未重新提交平台。

## 已验证的最快复现

```bash
git clone https://github.com/whitecat20250907/clip2-final.git
cd clip2-final
python -m venv .venv
# Linux: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install numpy==2.1.3 torch==2.14.1 --extra-index-url https://download.pytorch.org/whl/cpu
python reproduce.py
```

`reproduce.py` 从冻结 PF/四读出增量/LMY 特征接口和真实 13 参数适配器重新计算两套预测，核验逐字节 SHA。输出在 `generated/`。此路径不需要 GPU 或下载数据集；它是**缓存接口重放**，不是从原始音频重新提特征。

| 数据 | 行数 | 最终 RMSE | PCC |
|---|---:|---:|---:|
| VALID，联合，SID 587860 | 2117 | 15.294557340680246 | 0.8758093798569538 |
| EVAL，联合，T019 | 2189 | 16.2605239031252 | 0.8448856575229663 |
| VALID，本人分支，SID 587855 | 2117 | 15.494434050546618 | 0.8723208297249473 |

最终 VALID SHA256：`a061c982965dae202bb749121c67c085c7953df32d4305057e64dbc7f187d883`。
最终 EVAL SHA256：`0be40f574d231540f7efcf0802c2b9c464301d9518beec063caf9c96a42dd5da`。
VALID 归档保留原无表头格式；EVAL 保留最终交付格式。本仓库不自动提交。

## 下载完整归档

```bash
gh release download frozen-20261114 --repo whitecat20250907/clip2-final --dir assets
python verify_assets.py
```

Release 中保留冻结本机资产与 4090 原始依赖。压缩包内保留原路径映射，避免同名 `model.pt` 混淆。必要祖先模型不属于可以删除的旧最终版本。第三方数据与模型不新增授权许可。

## 原始音频推理与复现边界

`archive/final_code` 为最终融合/推理工程源码；`archive/scientific_ancestors` 为 SHA 锁定的原算法依赖；`archive/4090` 为服务器实际运行代码。Release 中包含微调适配器、冻结读出头与选定依赖。

这些原始脚本仍含旧服务器绝对路径、UID 和不同 Python 环境约束。**尚未在一台全新机器上验证“只下载公开模型和音频即可一键全链推理”**。不要把缓存重放成功理解成完整原始音频冷启动验收。依赖清单中的缺项和外部环境也不能补零或随意替换。

原模型使用 4090 完成 EVAL 推理；严格原 5090 数值重现未通过。原始 VALID 与 4090 回放是不同数值执行记录，不能冒充逐位相同。

## 算法身份

本人分支保留冻结 PRIOR_FIXED，融合 MF、Phi4-MM、MOSS、Qwen2.5-Omni 四个冻结读出；PF 只计一次。与 LMY 固定分支通过原 13 参数受限条件权重头融合。原比例、归一化、阈值和舍入边界不变。

MOSS 最终祖先为 G01、72 维、593 参数；早期交接文档的 G02/55 维描述已被实际推理验证否定。

## 外部资源

数据集官方入口：https://cadenzachallenge.org/docs/clip2/data/data 。原数据根应含 `metadata`、`valid/signals`、`eval/signals` 及对应 `unprocessed`。

`provenance/external_dependencies.json` 记录原模型/环境/数据引用；`environments/resolved_packages.json` 保存已恢复运行环境的下载 URL、精确版本与 SHA256。不要将多个分支强制装进一个未验证的最新版环境。

`provenance/` 保存资产 SHA、体积和实际验收结果。归档不包含 SSH 凭据、GitHub 密钥或自动提交入口。

