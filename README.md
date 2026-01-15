# UPT eval-ood 分支

- 未更改 UPT 结构
- 直接使用原始的 UPT 解决 OOD 任务

checkpoints/upt-r50-hicodet.pt 是 UPT 作者公开的模型参数

评估指令：

```shell
python main.py --eval --resume checkpoints/upt-r50-hicodet.pt
```

