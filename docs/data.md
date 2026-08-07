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

## 目录结构

从 [BIRD 官网](https://bird-bench.github.io/) 和 mini-dev 的官方仓库下载并解压，
最终应该长这样（`data/bird` 只是默认路径，可以用 `--root` 改）：

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
