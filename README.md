# DAPL-master (Multi-Drug + Regression)

本專案基於原始 DAPL 實作進行重構與延伸，主要目標是把單藥物任務升級為多藥物統一建模，並加入回歸任務能力。

原始專案來源：
- [weiba/DAPL](https://github.com/weiba/DAPL)

## 這個版本的核心改動

### 流程改版：A 流程 -> B 流程

原始 A 流程：
1. `pretrain.py`
2. `precontext.py`
3. `prototypical.py`

本專案 B 流程：
1. `A_pretrain.py`
2. `B_precontext.py`
3. `C_prototypical.py`

### 主要修改方向

1. 單一藥物對應單模型 -> 多藥物共用單模型  
   將原先以藥物為單位切分模型的設計，改為可在同一模型中學習多藥物特徵與任務訊號。

2. 新增回歸任務  
   除了原本偏分類/排序導向設定外，新增回歸輸出能力，用於預測連續值型目標。

## 檔案對應說明

- `A_pretrain.py`：新版預訓練階段（取代 `pretrain.py`）
- `B_precontext.py`：新版藥物上下文/特徵建構階段（取代 `precontext.py`）
- `C_prototypical.py`：新版任務訓練與推論階段（取代 `prototypical.py`）
- `data.py`：資料讀取與處理
- `tools/`：模型與資料處理輔助工具

## 建議執行流程

```bash
python A_pretrain.py
python B_precontext.py
python C_prototypical.py
```

若需重現舊版流程，可改用原始腳本：

```bash
python pretrain.py
python precontext.py
python prototypical.py
```

## 專案目的

此版本希望在臨床藥物反應預測場景中，透過：
- 多藥物共同學習提升模型泛化能力
- 回歸任務補足連續值預測需求

來擴展原始 DAPL 架構在實務資料上的可用性。

## 備註

- 本專案的 `.gitignore` 已忽略大型資料夾與模型輸出（例如 `drugmodels/`、`output_dir/`、`result/`）。
- 若先前已追蹤過這些路徑，請先使用 `git rm -r --cached <path>` 取消追蹤再提交。
