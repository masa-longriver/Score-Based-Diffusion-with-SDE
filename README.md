# Score-Based Diffusion with SDE

## 動作方法
'''python
python main.py ['データセット名']

# ex.
python main.py cifar10
'''

デフォルトでCIFAR10を使った学習ができるようになっています。  
新たなデータセットを試したい場合は、以下の手順を踏んでください。
- データセットのconfigを読み込むpythonファイルを作成
- data.pyの中でconfigを読み込むpythonファイルをimport
- data.pyのDatasetクラスを適宜変更

## 参考
> [score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch)