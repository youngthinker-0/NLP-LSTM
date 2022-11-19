import math
import torch
import torch.nn as nn
import torch.optim as optim
import give_valid_test

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


def make_batch(train_path, word2number_dict, batch_size, n_step):
    all_input_batch = []
    all_target_batch = []

    text = open(train_path, 'r', encoding='utf-8')  # open the file

    input_batch = []
    target_batch = []
    for sen in text:
        word = sen.strip().split(" ")  # space tokenizer

        if len(word) <= n_step:  # pad the sentence
            word = ["<pad>"] * (n_step + 1 - len(word)) + word

        for word_index in range(len(word) - n_step):
            input = [word2number_dict[n] for n in word[word_index:word_index + n_step]]  # create (1~n-1) as input
            target = word2number_dict[
                word[word_index + n_step]]  # create (n) as target, We usually call this 'casual language model'
            input_batch.append(input)
            target_batch.append(target)

            if len(input_batch) == batch_size:
                all_input_batch.append(input_batch)
                all_target_batch.append(target_batch)
                input_batch = []
                target_batch = []

    return all_input_batch, all_target_batch


def make_dict(train_path):
    text = open(train_path, 'r', encoding='utf-8')  # open the train file
    word_list = set()  # a set for making dict

    for line in text:
        line = line.strip().split(" ")
        word_list = word_list.union(set(line))

    word_list = list(sorted(word_list))  # set to list

    word2number_dict = {w: i + 2 for i, w in enumerate(word_list)}
    number2word_dict = {i + 2: w for i, w in enumerate(word_list)}

    # add the <pad> and <unk_word>
    word2number_dict["<pad>"] = 0
    number2word_dict[0] = "<pad>"
    word2number_dict["<unk_word>"] = 1
    number2word_dict[1] = "<unk_word>"

    return word2number_dict, number2word_dict


class TextLSTM_byMyself(nn.Module):
    def __init__(self):
        super(TextLSTM_byMyself, self).__init__()
        self.C = nn.Embedding(n_class, embedding_dim=emb_size)

        # The parameters of n layers
        #
        self.W_ii = [nn.Linear(emb_size, n_hidden, bias=False)] * num_layer
        self.W_hi = [nn.Linear(n_hidden, n_hidden, bias=False)] * num_layer
        self.b_i = [nn.Parameter(torch.ones([n_hidden]))] * num_layer

        self.W_if = [nn.Linear(emb_size, n_hidden, bias=False)] * num_layer
        self.W_hf = [nn.Linear(n_hidden, n_hidden, bias=False)] * num_layer
        self.b_f = [nn.Parameter(torch.ones([n_hidden]))] * num_layer

        self.W_ig = [nn.Linear(emb_size, n_hidden, bias=False)] * num_layer
        self.W_hg = [nn.Linear(n_hidden, n_hidden, bias=False)] * num_layer
        self.b_g = [nn.Parameter(torch.ones([n_hidden]))] * num_layer

        self.W_io = [nn.Linear(emb_size, n_hidden, bias=False)] * num_layer
        self.W_ho = [nn.Linear(n_hidden, n_hidden, bias=False)] * num_layer
        self.b_o = [nn.Parameter(torch.ones([n_hidden]))] * num_layer
        # 这里的 W 和 b 用于将第 i 层的 h_t 转化为 和最初的输入 x 一样的维度
        # 以便使用 lstm_cell 函数，将 h_t 转为下一层的 x 继续进行迭代
        self.W = [nn.Linear(n_hidden, emb_size, bias=False)] * num_layer
        self.b = [nn.Parameter(torch.ones([emb_size]))] * num_layer

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.W_final = nn.Linear(n_hidden, n_class, bias=False)
        self.b_final = nn.Parameter(torch.ones([n_class]))
        self.dropout = nn.Dropout()

    def forward(self, X):
        X = self.C(X)  # X: [batch_size, n_step, emb_size]
        X = X.transpose(0, 1)  # X : [n_step, batch_size, emb_size]
        sample_size = X.size()[1]

        h_0 = [torch.zeros([sample_size, n_hidden])] * num_layer
        c_0 = [torch.zeros([sample_size, n_hidden])] * num_layer
        h_t = h_0
        c_t = c_0
        for x_init in X:
            # 先跑第 1step 的 x ，我将 step 理解为 torch 文档中的时刻。然后继续跑 t = 2 时的x，一共有 5 个时刻 ( n_step = 5 )
            # 对于每一层的 x 都有迭代刷新，新的 x 会覆盖旧的 x ，x_init 为此时放入 lstm 第一层的数据
            # 对于每一层而言，旧时刻的 h_t 和 c_t 会被新时刻的 h_t 和 c_t 所覆盖
            x = x_init
            for layer_index in range(num_layer):
                # 首先跑第t时刻lstm的第0层，再跑lstm的第1层....
                # 在通过第0层后，x使用的是上一层传来的h_t(经过了一个nn.Linear变换，使其维度和原始的x一样)
                # 模仿官方文档写成了 lstm_cell 函数
                x, h_t[layer_index], c_t[layer_index] = self.lstm_cell(h_t[layer_index], c_t[layer_index], x,
                                                                       layer_index);

        model_output = self.W_final(h_t[num_layer - 1]) + self.b_final


        return model_output

    def lstm_cell(self, h_t, c_t, x, layer_index):  # 模仿官方文档写成 lstm_cell 函数
        # 以下内容参考自 torch.nn.LSTM 的计算过程
        i_t = self.sigmoid(self.W_ii[layer_index](x) + self.W_hi[layer_index](h_t) + self.b_i[layer_index])
        f_t = self.sigmoid(self.W_if[layer_index](x) + self.W_hf[layer_index](h_t) + self.b_f[layer_index])
        g_t = self.tanh(self.W_ig[layer_index](x) + self.W_hg[layer_index](h_t) + self.b_g[layer_index])
        o_t = self.sigmoid(self.W_io[layer_index](x) + self.W_ho[layer_index](h_t) + self.b_o[layer_index])
        c_t = torch.mul(f_t, c_t) + torch.mul(i_t, g_t)
        h_t = torch.mul(o_t, self.tanh(c_t))
        # 将 h_t 利用一个nn.linear转化为了下一层输入的 x
        x = self.W[layer_index](h_t) + self.b[layer_index]
        x = self.dropout(x)
        return x, h_t, c_t

class TextLSTM_byTorch(nn.Module):
    def __init__(self):
        super(TextLSTM_byTorch, self).__init__()
        self.C = nn.Embedding(n_class, embedding_dim=emb_size)
        self.lstm = nn.LSTM(input_size=emb_size, hidden_size=n_hidden, num_layers=2)
        self.W = nn.Linear(n_hidden, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))


    def forward(self, X):
        X = self.C(X)
        X = X.transpose(0, 1) # X : [n_step, batch_size, embeding size]
        outputs, hidden = self.lstm(X)
        # outputs : [n_step, batch_size, num_directions(=1) * n_hidden]
        # hidden : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        outputs = outputs[-1] # [batch_size, num_directions(=1) * n_hidden]
        model = self.W(outputs) + self.b # model : [batch_size, n_class]
        return model

def train_lstm():
    model = TextLSTM_byMyself()
    model.to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    # Training
    batch_number = len(all_input_batch)
    train_loss = []
    train_ppl = []
    for epoch in range(all_epoch):
        count_batch = 0
        for input_batch, target_batch in zip(all_input_batch, all_target_batch):
            optimizer.zero_grad()

            # input_batch : [batch_size, n_step, n_class]
            output = model(input_batch)

            # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
            loss = criterion(output, target_batch)
            ppl = math.exp(loss.item())
            if (count_batch + 1) % 50 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'Batch:', '%02d' % (count_batch + 1), f'/{batch_number}',
                      'lost =', '{:.6f}'.format(loss), 'ppl =', '{:.6}'.format(ppl))

            loss.backward()
            optimizer.step()

            count_batch += 1

        # valid after training one epoch
        all_valid_batch, all_valid_target = give_valid_test.give_valid(word2number_dict, n_step)
        all_valid_batch.to(device)
        all_valid_target.to(device)

        total_valid = len(all_valid_target) * 128
        with torch.no_grad():
            total_loss = 0
            count_loss = 0
            for valid_batch, valid_target in zip(all_valid_batch, all_valid_target):
                valid_output = model(valid_batch)
                valid_loss = criterion(valid_output, valid_target)
                total_loss += valid_loss.item()
                count_loss += 1

            print(f'Valid {total_valid} samples after epoch:', '%04d' % (epoch + 1), 'lost =',
                  '{:.6f}'.format(total_loss / count_loss),
                  'ppl =', '{:.6}'.format(math.exp(total_loss / count_loss)))
            train_loss.append(total_loss / count_loss)
            train_ppl.append(math.exp(total_loss / count_loss))
        with open("./train_loss.txt", 'w') as train_los:
            train_los.write(str(train_loss))

        with open("./train_ppl.txt", 'w') as train_pp:
            train_pp.write(str(train_ppl))
        if (epoch + 1) % save_checkpoint_epoch == 0:
            torch.save(model, f'models/lstm_model_epoch{epoch + 1}.ckpt')


def test_lstm(select_model_path):
    model = torch.load(select_model_path, map_location="cpu")  # load the selected model

    # load the test data
    all_test_batch, all_test_target = give_valid_test.give_test(word2number_dict, n_step)
    total_test = len(all_test_target) * 128
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    count_loss = 0
    for test_batch, test_target in zip(all_test_batch, all_test_target):
        test_output = model(test_batch)
        test_loss = criterion(test_output, test_target)
        total_loss += test_loss.item()
        count_loss += 1

    print(f"Test {total_test} samples with {select_model_path}……………………")
    print('lost =', '{:.6f}'.format(total_loss / count_loss),
          'ppl =', '{:.6}'.format(math.exp(total_loss / count_loss)))


if __name__ == '__main__':
    choice = 1
    n_step = 5  # number of cells(= number of Step)
    n_hidden = 5  # number of hidden units in one cell
    batch_size = 512  # batch size
    learn_rate = 0.001
    all_epoch = 100  # the all epoch for training
    emb_size = 128  # embeding size
    save_checkpoint_epoch = 10  # save a checkpoint per save_checkpoint_epoch epochs
    num_layer = 2  # The number of layers
    train_path = 'data/train.txt'  # the path of train dataset

    word2number_dict, number2word_dict = make_dict(train_path)  # use the make_dict function to make the dict
    print("The size of the dictionary is:", len(word2number_dict))

    n_class = len(word2number_dict)  # n_class (= dict size)

    all_input_batch, all_target_batch = make_batch(train_path, word2number_dict, batch_size, n_step)  # make the batch
    print("The number of the train batch is:", len(all_input_batch))

    all_input_batch = torch.LongTensor(all_input_batch).to(device)  # list to tensor
    all_target_batch = torch.LongTensor(all_target_batch).to(device)
    if choice == 0:
        print("\nTrain the LSTM……………………")
        train_lstm()
    else:
        print("\nTest the LSTM……………………")
        select_model_path = "models/lstm_model_epoch20.ckpt"
        test_lstm(select_model_path)
