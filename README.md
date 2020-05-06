# Pytorchによる“ヒントを与える線画着色”
Generative Adversarial Network(GAN)を用いて線画を自動着色します。GeneratorはResidual Blockを取り入れたUnetの構造をしており，アップサンプリングにpixel shuffleを用いています。学習はGeneratorをMAE損失によってプレトレーニングし，次にpix2pixと同様のadversarial損失とMSEによってGANの学習を行います。

## Requirement
```
pytorch
torchvision
numpy
Pillow  
tensorboard
tqdm
```

## Utils
以下のコマンドで実行できますが、プレトレーニングとGANによるトレーニングの実行は別々にしているので、コマンドライン引数で指定する必要があります。

**<注> データセットは何とか集めてください。**


### Pre-train
> python3 main.py

### Train
> python3 main.py --mode "train"

tensorboardによって損失の推移を確認できます。
> tensorboard --logdir = runs

## Result
![output](https://user-images.githubusercontent.com/49662875/81128718-eb065980-8f7c-11ea-8d4f-fac5e67123ba.png)

上が入力の線画とヒントの画像。下がGeneratorの生成画像。

## Reference
[1] Phillip Isola et.al., "Image-to-Image Translation with Conditional Adversarial Nets", arXiv:1611.07004, (2016).

[2] Yuanzheng Ci et.al., "User-Guided Deep Anime Line Art Colorization with Conditional Adversarial Networks", arXiv:1808.03240, (2018).

[3] 初心者がchainerで線画着色してみた。わりとできた。   
https://qiita.com/taizan/items/cf77fd37ec3a0bef5d9d

[4] ヒントを与える線画着色   
https://medium.com/@crosssceneofwindff/%E3%83%92%E3%83%B3%E3%83%88%E3%82%92%E4%B8%8E%E3%81%88%E3%82%8B%E7%B7%9A%E7%94%BB%E7%9D%80%E8%89%B2-759e1871f25b

[5] カラー画像から線画を作る[OpenCV & python]  
https://www.mathgram.xyz/entry/cv/contour

