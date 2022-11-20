#双层LSTM网络

在 main.py 中有参数 choice ，选择 0 为训练模型，选择 1 为进行测试

在 main.py 中有参数 num_layer 意味 LSTM 层数，实际上我实现的是多层 LSTM 网络，将 num_layer 设置为 2 即为双层 LSTM

在 main.py 中有 TextLSTM_byMyself 为我自己书写的 LSTM ， TextLSTM_byTorch 为 torch.nn.LSTM 

运行paint_loss.py可以绘制loss曲线

运行paint_ppl.py可以绘制ppl曲线

模型训练结果保存在models文件夹下，默认每隔 10epochs 保存一次
