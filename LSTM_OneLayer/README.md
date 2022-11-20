#单层LSTM网络

在 main.py 中有参数choice，选择0为训练模型，选择1为进行测试

在 main.py 中有 TextLSTM_byMyself 为我自己书写的LSTM， TextLSTM_byTorch 为 torch.nn.LSTM 

运行paint_loss.py可以绘制loss曲线

运行paint_ppl.py可以绘制ppl曲线

模型训练结果保存在models文件夹下，默认每隔 10epochs 保存一次
