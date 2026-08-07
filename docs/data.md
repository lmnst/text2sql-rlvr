# 数据准备

BIRD 的数据不进 Git（`.gitignore` 里 `data/` 已排除），需要自己下载后摆成下面的目录结构。
代码不假设某一个固定路径，`discover_split` 会在 `--root` 下递归找已知文件名，所以解压出来
是什么名字一般不用改。

## 需要的划分

| 划分 | 用途 | 何时下载 |
|---|---|---|
| Mini-Dev | 全链路先在这上面跑通，迭代快一个数量级 | 第一步就要 |
| dev | 最终评测，**只在最后用一次** | 出 baseline 之后 |
| train | SFT 数据 + RL 的 reward 来源，并从中切 val | 做 SFT 之前 |

划分纪律见 AGENTS.md「数据划分纪律」。简单说：训练期间只碰 train，dev 不参与任何选择。

## 下载

```bash
python scripts/download_bird.py --split mini_dev
```

脚本会下载、**校验压缩包完整性**、解压（BIRD 把数据库压缩包套在外层压缩包里，需要解两层）、
然后验证目录结构。dev 和 train 换 `--split` 即可，但**先只下 mini_dev**。

断线可以直接重跑，会续传。但续传接缝出错时，zip 的中央目录在文件末尾、仍然是好的，
所以 `ZipFile()` 能正常打开，只有解压到中间某个成员时才炸——这个坑很隐蔽，
所以脚本在解压前会对整个压缩包做一次 CRC 校验，发现损坏就丢弃重下一次。

从德国拉阿里云北京的 bucket 本来就容易断，如果反复失败：

```bash
python scripts/download_bird.py --split mini_dev --force
```

`--force` 直接丢掉已有文件从头下。还不行的话用 `--url` 指向任意镜像的直链，
或者走 HuggingFace 的 `birdsql/bird_mini_dev`（可配合 `HF_ENDPOINT=https://hf-mirror.com`）。

官方下载地址（脚本内置，列在这里备查）：

| 划分 | URL |
|---|---|
| mini_dev | `https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip` |
| dev | `https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip` |
| train | `https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip` |

HuggingFace 上也有镜像（`birdsql/bird_mini_dev`），国内如果 OSS 直连慢可以走那边。

## 目录结构

解压后应该长这样（`data/bird` 只是默认路径，可以用 `--root` 改）：

```text
data/bird/
  MINIDEV/
    mini_dev_sqlite.json
    dev_databases/
      <db_id>/
        <db_id>.sqlite
        database_description/*.csv
  dev_20240627/
    dev.json
    dev_databases/
      <db_id>/<db_id>.sqlite
  train/
    train.json
    train_databases/
      <db_id>/<db_id>.sqlite
```

注意：

- 只用 **SQLite** 版本。mini-dev 另外发布了 MySQL / PostgreSQL 版，第一版不支持，
  也不要提前加兼容代码。
- `database_description/` 里的 CSV 是列描述，可选。编码混杂（UTF-8 / cp1252 都有），
  加载时已经做了多编码回退。加进 prompt 会明显变长，默认关闭，想用加 `--descriptions`。

## 验证

```bash
python scripts/prepare_bird.py --root data/bird --split mini_dev --check-gold
```

它会打印实际样本数、涉及多少个库、缺哪些 `.sqlite`，并且**执行一遍全部 gold SQL**。
最后两行是接下来要一直盯着的两个数：

- `gold executable` —— gold 本身跑不通的题，模型不可能做对。这是 Execution Accuracy
  的上界，比 100% 低多少要心里有数。
- `gold empty` —— gold 返回空结果集的比例。官方 EX 用 `set(pred) == set(gold)` 比较，
  两个空集合恒等，所以这个比例就是**一条永远返回空的 SQL 能白拿的分数**。
  这个数字直接决定了后面 reward hacking 那一节有没有东西可写。

各划分的样本数以这条命令的输出为准，不要引用记忆里的数字。

## 拿不到数据时

`src/` 里的东西不依赖 BIRD：单元测试自己建临时 SQLite 库，所以

```bash
python -m pytest
```

在干净 checkout 上、没有数据也没有 GPU 的机器上应该全绿。
