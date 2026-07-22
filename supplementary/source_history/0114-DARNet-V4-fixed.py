# %% md
# # DARNet-V4 for power load forecasting
# 运行示例：python supplementary/source_history/0114-DARNet-V4-fixed.py --mode patch --seed 42
# %% md
# - [x] 实现
# - [x] CNN-RNN第一层无快速通道，之后层加identity快速通道
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from xgboost import XGBRegressor

# 控制开关：启用或禁用Patch功能
ENABLE_PATCH = False  # True: 启用Patch功能, False: 禁用Patch功能
PATCH_SIZE = 4  # Patch大小

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


# %%
def random_seed_set(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def extract_last_timestep_from_patched_inputs(inputs, original_feature_dim):
    """Recover the true last timestep features from a flattened patch input."""
    inferred_patch_size = inputs.shape[2] // original_feature_dim
    start = (inferred_patch_size - 1) * original_feature_dim
    end = start + original_feature_dim
    return inputs[:, -1:, start:end]


def apply_ot_inductive_bias(x, rho, target_index=6):
    """Apply the OT-specific differencing used by the baseline, patch-aware."""
    if ENABLE_PATCH:
        patch_feature_dim = x.shape[2]
        original_feature_dim = patch_feature_dim // PATCH_SIZE
        reshaped_x = x.reshape(x.shape[0], x.shape[1], PATCH_SIZE, original_feature_dim)
        features = reshaped_x[:, :, :, [i for i in range(original_feature_dim) if i != target_index]]
        values = reshaped_x[:, :, :, target_index]

        features = features.reshape(x.shape[0], x.shape[1] * PATCH_SIZE, original_feature_dim - 1)
        values = values.reshape(x.shape[0], x.shape[1] * PATCH_SIZE)

        values_mean = torch.mean(values, dim=1)
        values = torch.cat((values_mean.unsqueeze(1), values), dim=1)
        values = values[:, 1:] - rho * values[:, :-1]

        patched_features = features.reshape(x.shape[0], x.shape[1], PATCH_SIZE, original_feature_dim - 1)
        patched_values = values.reshape(x.shape[0], x.shape[1], PATCH_SIZE, 1)
        patched_input = torch.cat((patched_features, patched_values), dim=3)
        return patched_input.reshape(x.shape[0], x.shape[1], PATCH_SIZE * original_feature_dim)

    features_indices = [i for i in range(x.shape[2]) if i != target_index]
    features = x[:, :, features_indices]
    values = x[:, :, target_index]
    values_mean = torch.mean(values, dim=1)
    values = torch.cat((values_mean.unsqueeze(1), values), dim=1)
    values = values[:, 1:] - rho * values[:, :-1]
    return torch.cat((features, values.unsqueeze(2)), dim=2)
# %% md
# ## Fourier 季节性特征
# %%
def add_fourier_features(data):
    """
    在原始数据上新增4列周期特征：
    - sin_day = sin(2πt / 24)
    - cos_day = cos(2πt / 24)
    - sin_week = sin(2πt / 168)
    - cos_week = cos(2πt / 168)
    
    :param data: 原始数据 (DataFrame)
    :return: 添加了Fourier特征的数据
    """
    # 获取时间步索引
    t = np.arange(len(data))
    
    # 计算Fourier特征
    sin_day = np.sin(2 * np.pi * t / 24)
    cos_day = np.cos(2 * np.pi * t / 24)
    sin_week = np.sin(2 * np.pi * t / 168)
    cos_week = np.cos(2 * np.pi * t / 168)
    
    # 添加新特征到数据
    data_with_fourier = data.copy()
    data_with_fourier['sin_day'] = sin_day
    data_with_fourier['cos_day'] = cos_day
    data_with_fourier['sin_week'] = sin_week
    data_with_fourier['cos_week'] = cos_week
    
    return data_with_fourier


# %% md
# ## load data
# %%
def load_etth1_data(use_seasonality=False):
    """加载ETTh1数据并预处理"""
    url = './data/ETTh1.csv'
    data = pd.read_csv(url, sep=',', parse_dates=['date'], index_col='date')
    feature_columns = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    data = data[feature_columns]
    
    # 如果启用季节性特征，添加Fourier特征
    if use_seasonality:
        data = add_fourier_features(data)
        print(f"=== 季节性特征已启用，数据维度: {data.shape} ===")
    
    return data


# %% md
# ## normalization
# %%
def normalization(data):
    """
    data: original data with load
    return: normalized data, scaler of load
    """
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    scaler_y = MinMaxScaler()
    scaler_y.fit_transform(data[['OT']])
    return normalized_data, scaler, scaler_y


# %% md
# ## build supervised dataset
# %%
def series_to_supervise(data, seq_len, target_len, patch_size=1):
    """
    convert series data to supervised data
    :param data: original data
    :param seq_len: length of input sequence
    :param target_len: length of ouput sequence
    :param patch_size: size of each patch for slicing (default: 1, no patching)
    :return: return two ndarrays-- input and output in format suitable to feed to RNN
    """
    dim_0 = data.shape[0] - seq_len - target_len + 1
    dim_1 = data.shape[1]

    if patch_size > 1:
        # Apply patching: reshape data into patches
        new_seq_len = seq_len // patch_size

        # Create patched arrays
        x = np.zeros((dim_0, new_seq_len, dim_1 * patch_size))
        y = np.zeros((dim_0, target_len, dim_1))

        for i in range(dim_0):
            # Extract and patch input sequence
            input_seq = data[i:i + seq_len]
            # Reshape to (new_seq_len, patch_size, dim_1) then flatten last two dimensions
            patched_input = input_seq.reshape(new_seq_len, patch_size, dim_1)
            patched_input = patched_input.reshape(new_seq_len, dim_1 * patch_size)
            x[i] = patched_input

            # Extract target sequence (no patching for targets)
            y[i] = data[i + seq_len:i + seq_len + target_len]
    else:
        # Original implementation without patching
        x = np.zeros((dim_0, seq_len, dim_1))
        y = np.zeros((dim_0, target_len, dim_1))
        for i in range(dim_0):
            x[i] = data[i:i + seq_len]
            y[i] = data[i + seq_len:i + seq_len + target_len]

    print("supervised data: shape of x: {}, shape of y: {}".format(
        x.shape, y.shape))
    return x, y


# %% md
# ## slice data for patch
# %%
def slice_data(train_x, train_y, patch_size=4):
    """
    对训练数据进行切片处理，按patch_size切割
    :param train_x: 输入数据 (样本数, 序列长度, 特征数)
    :param train_y: 目标数据 (样本数, 预测长度, 特征数)
    :param patch_size: 切片大小
    :return: 切片后的数据
    """
    print(f"=== 开始数据切片，patch_size={patch_size} ===")

    # 检查序列长度是否可被patch_size整除
    seq_len = train_x.shape[1]
    target_len = train_y.shape[1]

    if seq_len % patch_size != 0 or target_len % patch_size != 0:
        raise ValueError(f"序列长度({seq_len})和预测长度({target_len})必须能被patch_size({patch_size})整除")

    # 计算新的序列长度
    new_seq_len = seq_len // patch_size

    # 获取维度信息
    num_samples = train_x.shape[0]
    num_features = train_x.shape[2]

    # 创建切片后的数组
    sliced_train_x = np.zeros((num_samples, new_seq_len, num_features * patch_size))

    # 对每个样本进行切片处理
    for i in range(num_samples):
        # 切片输入数据
        input_seq = train_x[i]  # (seq_len, num_features)
        # 重塑为 (new_seq_len, patch_size, num_features) 然后展平最后两个维度
        patched_input = input_seq.reshape(new_seq_len, patch_size, num_features)
        patched_input = patched_input.reshape(new_seq_len, num_features * patch_size)
        sliced_train_x[i] = patched_input

    print(f"切片前数据形状: train_x={train_x.shape}, train_y={train_y.shape}")
    print(f"切片后数据形状: train_x={sliced_train_x.shape}, train_y={train_y.shape}")
    print("=== 数据切片完成 ===")

    return sliced_train_x, train_y


# %% md
# ## 5-folds TimeSeriesSplit
# %%
def time_series_split(X, Y, n_split=5):
    """
    X: features, size * seq_len * feature_num
    Y: labels, size * target_len
    return: list of train_x, test_x, train_y, test_y
    """
    tscv = TimeSeriesSplit(n_splits=n_split)
    train_x_list = list()
    valid_x_list = list()
    train_y_list = list()
    valid_y_list = list()
    for train_index, valid_index in tscv.split(X):
        train_x_list.append(X[train_index])
        train_y_list.append(Y[train_index])
        valid_x_list.append(X[valid_index])
        valid_y_list.append(Y[valid_index])
    return train_x_list, train_y_list, valid_x_list, valid_y_list


# %% md
# ## DARNet model
# %% md
# ### CNN1D-RNN Block
# %%
class CNN1D_RNN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, kernel_size=3):
        super(CNN1D_RNN_Block, self).__init__()
        # params
        padding = int(kernel_size / 2)
        self.in_channels = in_channels
        self.out_channels = out_channels

        # layers
        self.CNN = nn.Conv1d(in_channels,
                             out_channels,
                             kernel_size,
                             padding=padding)
        self.RNN1 = nn.GRU(input_size=out_channels,
                           hidden_size=out_channels,
                           num_layers=2,
                           dropout=dropout,
                           batch_first=True)
        self.RNN2 = nn.GRU(input_size=in_channels,
                           hidden_size=out_channels,
                           num_layers=2,
                           dropout=dropout,
                           batch_first=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        '''
        x shape (batch_size, seq_len, d_feature)
        '''
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        cnn_input = x.permute(0, 2, 1)
        # cnn_input shape (batch_size, in_channel, seq_len)
        cnn_out = self.CNN(cnn_input)
        # cnn_out = self.bn1(cnn_out)
        cnn_out = self.relu(cnn_out)
        # cnn_out shape (batch_size, out_channel, seq_len)

        rnn_input = cnn_out.permute(0, 2, 1)
        # rnn_input shape (batch_size, seq_len, out_channels)
        rnn_out_1, _ = self.RNN1(rnn_input)
        # rnn_out_1 shape (batch_size, seq_len, out_channels)

        if self.in_channels != self.out_channels:
            out = self.relu(rnn_out_1)
        else:
            out = self.relu(rnn_out_1 + x)

        return out


# %% md
# ### RNN-CNN2D Block
# %%
class RNN_CNN2D_Block(nn.Module):
    def __init__(self, input_size, out_channels, seq_len, dropout, n_layers=2):
        super(RNN_CNN2D_Block, self).__init__()

        # layers
        self.RNN = nn.GRU(input_size=input_size,
                          hidden_size=out_channels,
                          num_layers=n_layers,
                          dropout=dropout,
                          batch_first=True)
        self.CNN = nn.Conv2d(in_channels=1,
                             out_channels=out_channels,
                             kernel_size=(seq_len, 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        '''
        x shape (batch_size, seq_len, input_size)
        '''
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        rnn_out, hidden_state = self.RNN(x)
        # rnn_out shape (batch_size, seq_len, d_features)

        cnn_input = rnn_out.unsqueeze(1)
        # cnn_input shape (batch_size, 1, seq_len, d_featues)
        cnn_out = self.CNN(cnn_input)
        # cnn_out shape (batch_size, out_channels, 1, d_features)
        cnn_out = cnn_out.squeeze(2).permute(0, 2, 1)
        # cnn_out = self.bn2(cnn_out)
        cnn_out = self.relu(cnn_out)

        # cnn_out shape (batch_size, d_features, out_channels)

        return cnn_out, hidden_state


# %% md
# ### Encoder
# %%
class Encoder(nn.Module):
    def __init__(self, input_size, num_channels, seq_len, dropout=0.5):
        super(Encoder, self).__init__()
        '''
        input_size(int): dimension of features
        num_channels(list): channels of each cnn-rnn layer
        seq_len(int): window length of input
        '''
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [CNN1D_RNN_Block(in_channels, out_channels, dropout)]

        self.cnn_rnn = nn.Sequential(*layers)
        self.rnn_cnn = RNN_CNN2D_Block(input_size, num_channels[-1], seq_len, dropout)

def forward(self, x, rho, target_index=6):
        '''
        x shape (batch_size, seq_len, input_size)
        '''
        inp = apply_ot_inductive_bias(x, rho, target_index=target_index)
        cnn_rnn_out = self.cnn_rnn(inp)
        rnn_cnn_out, hidden_state = self.rnn_cnn(inp)
        return cnn_rnn_out, rnn_cnn_out, hidden_state

        # 根据ENABLE_PATCH判断处理逻辑
        if ENABLE_PATCH:
            # Patch模式：直接使用整个输入，不做OT列分离和差分处理
            # 因为patch后输入已经被flatten，target_index=6不再对应OT列
            inp = x  # 直接使用整个输入
        else:
            # 非Patch模式：保持原来的逻辑，分离OT列并做差分处理
            # 分离特征和目标值（OT列）
            features_indices = [i for i in range(x.shape[2]) if i != target_index]
            target_indices = [target_index]
            
            features = x[:, :, features_indices]
            values = x[:, :, target_indices].squeeze(-1)  # 移除最后一个维度
            # features shape (batch_size, num_steps, input_size -1)
            # values shape (batch_size, num_steps)
            values_mean = torch.mean(values, dim=1)
            # values_mean shape (batch_size)
            values = torch.cat((values_mean.unsqueeze(1), values), dim=1)
            # values shape (batch_size, num_steps + 1)
            values = values[:, 1:] - rho * values[:, :-1]
            # values shape (batch_size, num_steps)
            inp = torch.cat((features, values.unsqueeze(2)), dim=2)
            # inp shape (batch_size, num_steps, input_size)

        cnn_rnn_out = self.cnn_rnn(inp)
        # cnn_rnn_out shape (batch_size, seq_len, num_channels[-1])

        rnn_cnn_out, hidden_state = self.rnn_cnn(inp)
        # rnn_cnn_out shape (batch_size, num_channels[-1], num_channels[-1])

        return cnn_rnn_out, rnn_cnn_out, hidden_state


# %% md
# ### Attention
# %%
class AdditiveAttention(nn.Module):
    """加性注意力"""

    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 在维度扩展后，
        # `queries` 的形状：(`batch_size`，查询的个数，1，`num_hidden`)
        # `key` 的形状：(`batch_size`，1，“键－值”对的个数，`num_hiddens`)
        # 使用广播方式进行求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # `self.w_v` 仅有一个输出，因此从形状中移除最后那个维度。
        # `scores` 的形状：(`batch_size`，查询的个数，“键-值”对的个数)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = nn.functional.softmax(scores, dim=-1)
        # `values` 的形状：(`batch_size`，“键－值”对的个数，值的维度)
        return torch.bmm(self.dropout(self.attention_weights), values)


class DotProductAttention(nn.Module):
    """缩放点积注意力"""

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # `queries` 的形状：(`batch_size`，查询的个数，`d`)
    # `keys` 的形状：(`batch_size`，“键－值”对的个数，`d`)
    # `values` 的形状：(`batch_size`，“键－值”对的个数，值的维度)
    # `valid_lens` 的形状: (`batch_size`，) 或者 (`batch_size`，查询的个数)
    def forward(self, queries, keys, values):
        d = queries.shape[-1]
        # 设置 `transpose_b=True` 为了交换 `keys` 的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = nn.functional.softmax(scores, dim=-1)
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(nn.Module):
    """多头注意力"""

    def __init__(self,
                 key_size,
                 query_size,
                 value_size,
                 num_hiddens,
                 num_heads,
                 dropout,
                 bias=False,
                 **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens=None):
        # `queries`，`keys`，`values` 的形状:
        # (`batch_size`，查询或者“键－值”对的个数，`num_hiddens`)
        # `valid_lens`　的形状:
        # (`batch_size`，) 或 (`batch_size`，查询的个数)
        # 经过变换后，输出的 `queries`，`keys`，`values`　的形状:
        # (`batch_size` * `num_heads`，查询或者“键－值”对的个数，
        # `num_hiddens` / `num_heads`)
        queries = transpose_qkv(queries, self.num_heads)
        keys = transpose_qkv(keys, self.num_heads)
        values = transpose_qkv(values, self.num_heads)

        if valid_lens is not None:
            # 在轴 0，将第一项（标量或者矢量）复制 `num_heads` 次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(valid_lens,
                                                 repeats=self.num_heads,
                                                 dim=0)

        # `output` 的形状: (`batch_size` * `num_heads`，查询的个数，
        # `num_hiddens` / `num_heads`)
        output = self.attention(queries, keys, values)

        # `output_concat` 的形状: (`batch_size`，查询的个数，`num_hiddens`)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状。"""
    # 输入 `X` 的形状: (`batch_size`，查询或者“键－值”对的个数，`num_hiddens`)
    # 输出 `X` 的形状: (`batch_size`，查询或者“键－值”对的个数，`num_heads`，
    # `num_hiddens` / `num_heads`)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出 `X` 的形状: (`batch_size`，`num_heads`，查询或者“键－值”对的个数,
    # `num_hiddens` / `num_heads`)
    X = X.permute(0, 2, 1, 3)

    # 最终输出的形状: (`batch_size` * `num_heads`, 查询或者“键－值”对的个数,
    # `num_hiddens` / `num_heads`)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """逆转 `transpose_qkv` 函数的操作。"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


# %% md
# ### RNN-Attn Block
# %%
class RNN_Attn_Block(nn.Module):
    def __init__(self, input_size, hidden_dim, i, dropout=0.5, use_attention=True):
        super(RNN_Attn_Block, self).__init__()
        # params
        self.i = i
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention

        # 简化处理：统一使用标准输入大小
        rnn_input_size = input_size + hidden_dim

        # layers
        self.RNN = nn.GRU(input_size=rnn_input_size,
                          hidden_size=hidden_dim,
                          num_layers=2,
                          batch_first=True,
                          dropout=dropout)
        # self.attention1 = DotProductAttention(dropout)
        # self.attention2 = DotProductAttention(dropout)
        self.attention1 = AdditiveAttention(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.attention2 = AdditiveAttention(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.relu = nn.ReLU()
        self.dense = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, x, state):
        '''
        x shape (batch_size, 1, input_size)
        state[0] cnn_rnn_out shape (batch_size, 72, hidden_dim) 或 (batch_size, 18, hidden_dim) 在Patch模式下
        state[1] rnn_cnn_out shape (batch_size, hidden_dim, hidden_dim)
        state[2][i] shape (num_layers, batch_size, hidden_dim)
        '''
        cnn_rnn_out = state[0]
        rnn_cnn_out = state[1]

        # use_attention开关：如果为False，直接使用RNN输出，不进行attention计算
        if not self.use_attention:
            # 不使用attention，但需要保持输入维度一致
            # 创建一个与attention路径相同的输入维度
            dummy_context = torch.zeros(x.shape[0], 1, self.hidden_dim, device=x.device)
            rnn_input = torch.cat((x, dummy_context), dim=-1)
            
            # 不使用attention，直接进行RNN计算
            rnn_out, state[2][self.i] = self.RNN(rnn_input, state[2][self.i])
            # rnn_out shape (batch_size, 1, hidden_dim)
            # state shape (num_layers, batch_size, hidden_dim)
            
            # 不使用attention，直接返回RNN输出
            out = rnn_out
            if self.input_size != self.hidden_dim:
                out = self.relu(out + rnn_out)
            # out shape (batch_size, 1, hidden_dim)
            else:
                out = self.relu(out + x)
            
            return out, state
        
        # 如果use_attention为True，保持原有attention逻辑
        query = state[2][self.i][-1].unsqueeze(1)
        # query shape (batch_size, 1, hidden_dim)

        context_1 = self.attention1(query, cnn_rnn_out, cnn_rnn_out)
        # context_1 shape (batch_size, 1, hidden_dim)

        # 统一处理输入
        rnn_input = torch.cat((x, context_1), dim=-1)

        rnn_out, state[2][self.i] = self.RNN(rnn_input, state[2][self.i])
        # rnn_out shape (batch_size, 1, hidden_dim)
        # state shape (num_layers, batch_size, hidden_dim)

        context_2 = self.attention2(rnn_out, rnn_cnn_out, rnn_cnn_out)
        # context_2 shape (batch_size, 1, hidden_dim)

        out = self.dense(torch.cat((rnn_out, context_2), dim=-1))
        if self.input_size != self.hidden_dim:
            out = self.relu(out + rnn_out)
        # out shape (batch_size, 1, hidden_dim)
        else:
            out = self.relu(out + x)

        return out, state


# %% md
# ### Decoder
# %%
class Decoder(nn.Module):
    def __init__(self, input_size, num_hidden_dim, dropout, use_attention=True):
        super(Decoder, self).__init__()
        layers = []
        dense_layers = []
        num_levels = len(num_hidden_dim)

        for i in range(num_levels):
            in_size = input_size if i == 0 else num_hidden_dim[i - 1]
            out_size = num_hidden_dim[i]
            layers += [RNN_Attn_Block(in_size, out_size, i, dropout, use_attention=use_attention)]

        input_dim = num_hidden_dim[-1]
        while (input_dim > 4):
            dense_layers += [
                nn.Linear(input_dim, round(input_dim / 2)),
                nn.ReLU()
            ]
            input_dim = round(input_dim / 2)

        dense_layers += [nn.Linear(input_dim, 1)]

        self.blks = nn.Sequential(*layers)
        self.dense = nn.Sequential(*dense_layers)

    def forward(self, x, state):
        '''
        x shape (batch_size, 1, input_size)
        state[0] cnn_rnn_out shape (batch_size, 72, hidden_dim)
        state[1] rnn_cnn_out shape (batch_size, hidden_dim, hidden_dim)
        state[2] shape n * (num_layers, batch_size, hidden_dim)
        '''
        for i, blk in enumerate(self.blks):
            x, state = blk(x, state)

        out = self.dense(x)
        # out shape (batch_size, 1, 1)
        return out, state


# %% md
# ### DARNet
# %%
class DARNet(nn.Module):
    def __init__(self,
                 input_size,
                 num_channels,
                 seq_len,
                 num_hidden_dim,
                 dropout=0.5,
                 use_attention=False):
        super(DARNet, self).__init__()
        # params
        self.num_layers = len(num_hidden_dim)
        self.use_attention = use_attention

        # 简化处理：在Patch模式下，只调整编码器的输入维度
        # 解码器保持原始输入维度，避免复杂的维度调整
        if ENABLE_PATCH:
            # 编码器使用Patch后的维度
            encoder_input_size = input_size * PATCH_SIZE
            encoder_seq_len = seq_len // PATCH_SIZE
            # 解码器保持原始维度
            decoder_input_size = input_size
            print(f"=== Patch模式: 编码器输入维度={encoder_input_size}, 序列长度={encoder_seq_len} ===")
            print(f"=== Patch模式: 解码器保持原始维度 input_size={decoder_input_size} ===")
        else:
            encoder_input_size = input_size
            encoder_seq_len = seq_len
            decoder_input_size = input_size
            print("=== 标准模式: 使用原始输入维度和序列长度 ===")

        # layers
        self.encoder = Encoder(encoder_input_size, num_channels, encoder_seq_len, dropout)
        self.decoder = Decoder(decoder_input_size, num_hidden_dim, dropout, use_attention=use_attention)

        # 打印attention开关状态
        print(f"=== Attention开关状态: use_attention={use_attention} ===")

    def forward(self, enc_inputs, dec_inputs, target_index=6):
        '''
        enc_inputs shape (batch_size, seq_len, input_size)
        dec_inputs shape (batch_size, tar_len, input_size)
        '''
        rho = torch.ones(1, device=device)
        y_ = dec_inputs[:, :1, target_index].clone()  # 使用target_index而不是-1
        # y_ shape (batch_size, 1)
        dec_inputs[:, :1, target_index] = dec_inputs[:, :1, target_index] - rho * torch.mean(
            dec_inputs[:, :, target_index], dim=-1, keepdim=True)

        cnn_rnn_out, rnn_cnn_out, hidden_state = self.encoder(enc_inputs, rho, target_index=target_index)

        # 保持原有逻辑：使用最后一个时间步的hidden state
        context = hidden_state[:, -1, :]  # (batch, hidden_dim)

        state = [cnn_rnn_out, rnn_cnn_out, [hidden_state] * self.num_layers]

        outputs = []

        for i in range(dec_inputs.shape[1]):
            if i:
                # 复制当前时间步的所有特征，只替换target_index对应的列为模型预测值
                x = dec_inputs[:, i:i + 1, :].clone()
                x[:, :, target_index:target_index+1] = out.detach()  # 只替换目标列，保持维度一致
            else:
                x = dec_inputs[:, i:i + 1, :]
                # x shape (batch_size, 1, input_size)
            out, state = self.decoder(x, state)
            # out shape (batch_size, 1, 1)
            outputs.append(out)
        outputs = torch.cat(outputs, dim=1).squeeze(-1)
        outputs = outputs + rho * y_
        # outputs shape (batch_size, 24)
        return outputs


#%% md
# ### Attention Layer (已删除，使用RNN_Attn_Block内部的attention开关)
#%%


# %% md
# ### test model
# %%
# model = DARNet(16, [32, 64, 64], 72, [64, 64, 64]).to(device)
# x_1 = torch.randn(10, 72, 16).to(device)
# x_2 = torch.randn(10, 24, 16).to(device)
# out = model(x_1, x_2)
# out.shape
# %% md
# ## lr-scheduler
# %%
class SchedulerCosineDecayWarmup:
    def __init__(self, optimizer, lr, warmup_len, total_iters):
        self.optimizer = optimizer
        self.lr = lr
        self.warmup_len = warmup_len
        self.total_iters = total_iters
        self.current_iter = 0

    def get_lr(self):
        if self.current_iter < self.warmup_len:
            lr = self.lr * (self.current_iter + 1) / self.warmup_len
        else:
            cur = self.current_iter - self.warmup_len
            total = self.total_iters - self.warmup_len
            lr = 0.1 * (1 + 9 * np.cos(np.pi * cur / total)) * self.lr
        return lr

    def step(self):
        lr = self.get_lr()
        for param in self.optimizer.param_groups:
            param['lr'] = lr
        self.current_iter += 1


# %% md
# ## model training for HPO
# %%
def train_model_hpo(train_x_list, train_y_list, valid_x_list, valid_y_list,
                    input_size, seq_len, target_len, mse_thresh, hidden_dim,
                    n_layers, number_epoch, batch_size, lr, drop_prob,
                    weight_decay, target_index=6):
    valid_loss_list = []
    for num in range(len(train_x_list)):
        while (1):
            model = DARNet(input_size, [hidden_dim] * n_layers, seq_len, [hidden_dim] * n_layers, drop_prob)
            model = model.to(device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=lr,
                                         weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                        1,
                                                        gamma=0.98)
            valid_loss_min = np.inf
            print('train dataset {}'.format(num))
            train_x = train_x_list[num]
            train_y = train_y_list[num]
            valid_x = valid_x_list[num]
            valid_y = valid_y_list[num]
            train_dataset = TensorDataset(torch.FloatTensor(train_x),
                                          torch.FloatTensor(train_y))
            valid_dataset = TensorDataset(torch.FloatTensor(valid_x),
                                          torch.FloatTensor(valid_y))

            train_loader = DataLoader(dataset=train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      drop_last=True)
            valid_loader = DataLoader(dataset=valid_dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      drop_last=True)
            train_losses = list()

            num_without_imp = 0

            # train
            for epoch in range(1, number_epoch + 1):
                loop = tqdm(enumerate(train_loader),
                            total=len(train_loader),
                            leave=True,
                            ncols=100)
                for i, (inputs, labels) in loop:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    encoder_inputs = inputs
                    decoder_inputs = torch.cat(
                        (inputs[:, -1:, :], labels[:, :-1, :]), dim=1)
                    outputs = model(encoder_inputs, decoder_inputs)
                    loss = criterion(outputs, labels[:, :, target_index])
                    train_losses.append(loss.item)
                    loss.backward()
                    optimizer.step()

                    # eval
                    if i % 5 == 0:
                        num_without_imp = num_without_imp + 1
                        valid_losses = list()
                        model.eval()
                        for inp, lab in valid_loader:
                            inp = inp.to(device)
                            lab = lab.to(device)
                            encoder_inp = inp
                            decoder_inp = torch.cat(
                                (inp[:, -1:, :], lab[:, :-1, :]), dim=1)
                            out = model(encoder_inp, decoder_inp)
                            valid_loss = criterion(out, lab[:, :, target_index])
                            valid_losses.append(valid_loss.item())
                        model.train()
                        loop.set_description("Epoch: {}/{}...".format(
                            epoch, number_epoch))
                        loop.set_postfix(train_loss=loss.item(),
                                         valid_loss=np.mean(valid_losses))
                        if np.mean(valid_losses) < valid_loss_min:
                            num_without_imp = 0
                            valid_loss_min = np.mean(valid_losses)
                if num_without_imp > 50:
                    pass

                #                     break
                scheduler.step()
            if valid_loss_min < mse_thresh:
                valid_loss_list.append(valid_loss_min)
                break
    return np.mean(valid_loss_list)


# %% md
# ## hyper-parameters config
# %%
seq_len = 72
target_len = 24
mse_thresh = 0.05


def model_config():
    batch_sizes = [256, 512]
    lrs = [0.01]
    number_epochs = [40]
    hidden_dims = [64, 128]
    n_layers = [2, 3]
    drop_prob = [0]
    weight_decays = [0]
    configs = list()
    for i in batch_sizes:
        for j in lrs:
            for k in number_epochs:
                for l in hidden_dims:
                    for m in n_layers:
                        for n in drop_prob:
                            for o in weight_decays:
                                configs.append({
                                    'batch_size': i,
                                    'lr': j,
                                    'number_epoch': k,
                                    'hidden_dim': l,
                                    'n_layers': m,
                                    'drop_prob': n,
                                    'weight_decay': o
                                })
    return configs


# %% md
# ## random search for HPO
# %%
def run_model_hpo(seq_len=seq_len,
                  target_len=target_len,
                  mse_thresh=mse_thresh):
    # 使用新的数据加载函数
    data = load_etth1_data()
    train_data = data[:int(0.8 * len(data))]
    train_data, _, _ = normalization(train_data)
    train_x, train_y = series_to_supervise(train_data, seq_len, target_len)
    train_x_list, train_y_list, valid_x_list, valid_y_list = time_series_split(
        train_x, train_y)
    #         with enough data
    train_x_list = train_x_list[-1:]
    train_y_list = train_y_list[-1:]
    valid_x_list = valid_x_list[-1:]
    valid_y_list = valid_y_list[-1:]

    configs = model_config()
    records = []
    input_size = train_x.shape[2]
    for i in range(6):
        config = random.choice(configs)
        configs.remove(config)
        batch_size = config['batch_size']
        lr = config['lr']
        number_epoch = config['number_epoch']
        hidden_dim = config['hidden_dim']
        n_layers = config['n_layers']
        drop_prob = config['drop_prob']
        weight_decay = config['weight_decay']
        print(
            "model config: batch_size-{}, lr-{}, number_epoch-{}, hidden_dim-{}, n_layers-{},drop_prob-{},weight_decay-{}"
            .format(batch_size, lr, number_epoch, hidden_dim, n_layers,
                    drop_prob, weight_decay))
        valid_loss = train_model_hpo(
            train_x_list,
            train_y_list,
            valid_x_list,
            valid_y_list,
            input_size,
            seq_len,
            target_len,
            mse_thresh,
            hidden_dim,
            n_layers,
            number_epoch,
            batch_size,
            lr,
            drop_prob,
            weight_decay,
        )
        records.append({
            'batch_size': batch_size,
            'lr': lr,
            'number_epoch': number_epoch,
            'hidden_dim': hidden_dim,
            'n_layers': n_layers,
            'drop_prob': drop_prob,
            'weight_decay': weight_decay,
            'valid_loss': valid_loss
        })
    return records


# %% md
# ## RUN random search
# %%
# random_seed_set(42)
# records = run_model_hpo()  # 注释掉HPO，只运行baseline
# %% md
# ## find the best hyper-parameters
# %%
# records = pd.DataFrame(records).sort_values(by='valid_loss')
# records.to_csv('./records/DARNet_records.csv', mode='a', index=False, header=False)
# records
# %% md
# ## retrain a model
# %%
def train_model(train_x, train_y, valid_x, valid_y, input_size, seq_len,
                target_len, mse_thresh, hidden_dim, n_layers, number_epoch,
                batch_size, lr, drop_prob, weight_decay, save_path="./checkpoints/DARNetV4_state_dict.pt", patch_size=1,
                use_attention=False, target_index=6):
    # 根据patch_size进行数据切片
    if patch_size > 1:
        print(f"=== 在train_model内部进行数据切片，patch_size={patch_size} ===")
        train_x, train_y = slice_data(train_x, train_y, patch_size)
        valid_x, valid_y = slice_data(valid_x, valid_y, patch_size)
        print(f"切片后数据形状: train_x={train_x.shape}, valid_x={valid_x.shape}")

    # 模型初始化使用传入参数
    model = DARNet(
        input_size,
        [hidden_dim] * n_layers,
        seq_len,
        [hidden_dim] * n_layers,
        dropout=drop_prob,
        use_attention=use_attention
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.98)

    valid_loss_min = np.inf
    train_dataset = TensorDataset(torch.FloatTensor(train_x),
                                  torch.FloatTensor(train_y))
    valid_dataset = TensorDataset(torch.FloatTensor(valid_x),
                                  torch.FloatTensor(valid_y))

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              drop_last=True)

    train_loss_list = []
    valid_loss_list = []

    # 训练一次，不依赖mse_thresh来break
    for epoch in range(1, number_epoch + 1):
        loop = tqdm(enumerate(train_loader),
                    total=len(train_loader),
                    leave=True,
                    ncols=100)
        for i, (inputs, labels) in loop:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            encoder_inputs = inputs

            # 根据Patch模式调整解码器输入
            if ENABLE_PATCH:
                # Patch模式: inputs的维度是 (batch_size, 18, 44), labels的维度是 (batch_size, 24, original_feature_dim)
                # 解码器需要原始维度的输入，所以需要从Patch数据中提取原始特征
                # 取inputs的最后时间步，并提取前original_feature_dim个特征（与labels维度匹配）
                original_feature_dim = labels.shape[2]  # 动态获取原始特征维度
                last_input = extract_last_timestep_from_patched_inputs(inputs, original_feature_dim)
                decoder_inputs = torch.cat((last_input, labels[:, :-1, :]), dim=1)
            else:
                # 标准模式: inputs和labels的特征维度相同
                decoder_inputs = torch.cat((inputs[:, -1:, :], labels[:, :-1, :]), dim=1)

            outputs = model(encoder_inputs, decoder_inputs, target_index=target_index)
            loss = criterion(outputs, labels[:, :, target_index])
            loss.backward()
            optimizer.step()

            # eval
            if i % 5 == 0:
                valid_losses = list()
                model.eval()
                for inp, lab in valid_loader:
                    inp = inp.to(device)
                    lab = lab.to(device)
                    encoder_inp = inp

                    # 根据Patch模式调整解码器输入
                    if ENABLE_PATCH:
                        # Patch模式: inp的维度是 (batch_size, 18, 44), lab的维度是 (batch_size, 24, original_feature_dim)
                        original_feature_dim = lab.shape[2]  # 动态获取原始特征维度
                        last_input = extract_last_timestep_from_patched_inputs(inp, original_feature_dim)
                        decoder_inp = torch.cat((last_input, lab[:, :-1, :]), dim=1)
                    else:
                        # 标准模式: inp和lab的特征维度相同
                        decoder_inp = torch.cat((inp[:, -1:, :], lab[:, :-1, :]), dim=1)

                    out = model(encoder_inp, decoder_inp, target_index=target_index)
                    valid_loss = criterion(out, lab[:, :, target_index])
                    valid_losses.append(valid_loss.item())
                model.train()
                loop.set_description("Epoch: {}/{}...".format(
                    epoch, number_epoch))
                loop.set_postfix(train_loss=loss.item(),
                                 valid_loss=np.mean(valid_losses))
                train_loss_list.append(loss.item())
                valid_loss_list.append(np.mean(valid_losses))

                # 保存best模型逻辑保留
                if np.mean(valid_losses) < valid_loss_min:
                    torch.save(model.state_dict(), save_path)
                    valid_loss_min = np.mean(valid_losses)
        scheduler.step()

    return model, train_loss_list, valid_loss_list


# %% md
# ## test results
# %%
def test_model(model, test_x, test_y, scaler_y, seq_len, target_len,
               batch_size, model_path="./checkpoints/DARNetV4_state_dict.pt", patch_size=1, target_index=6):
    # 根据patch_size进行数据切片（与train_model保持一致）
    if patch_size > 1:
        print(f"=== 在test_model内部进行数据切片，patch_size={patch_size} ===")
        test_x, test_y = slice_data(test_x, test_y, patch_size)
        print(f"测试数据切片后形状: test_x={test_x.shape}, test_y={test_y.shape}")
    
    test_dataset = TensorDataset(torch.FloatTensor(test_x),
                                 torch.FloatTensor(test_y))
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             drop_last=False)
    model.load_state_dict(torch.load(model_path))
    y_pred = []
    y_true = []
    with torch.no_grad():
        model.eval()
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            encoder_inputs = inputs
            
            encoder_inputs = inputs

            # 初始 decoder 输入：最后一个已知时间步
            if patch_size > 1:
                original_feature_dim = labels.shape[2]
                decoder_step = extract_last_timestep_from_patched_inputs(inputs, original_feature_dim).clone()
            else:
                decoder_step = inputs[:, -1:, :].clone()

            pred_steps = []

            for t in range(target_len):
                outputs = model(encoder_inputs, decoder_step, target_index=target_index)

                # 取当前步预测
                if outputs.dim() == 2:
                    current_pred = outputs[:, -1:].unsqueeze(1)   # [batch, 1, 1]
                else:
                    current_pred = outputs[:, -1:, 0:1]            # [batch, 1, 1]

                pred_steps.append(current_pred.squeeze(-1))        # [batch, 1]

                # 还没到最后一步，就构造下一步输入
                if t < target_len - 1:
                    next_step = labels[:, t:t+1, :].clone()
                    next_step[:, :, target_index:target_index+1] = current_pred
                    decoder_step = torch.cat((decoder_step, next_step), dim=1)

            outputs = torch.cat(pred_steps, dim=1)
            y_pred += outputs.cpu().numpy().reshape(-1).tolist()
            y_true += labels[:, :, target_index].cpu().numpy().reshape(-1).tolist()
    y_pred = np.array(y_pred).reshape(-1, 1)
    y_true = np.array(y_true).reshape(-1, 1)
    #     pdb.set_trace()
    load_pred = scaler_y.inverse_transform(y_pred)
    load_true = scaler_y.inverse_transform(y_true)
    mean_pred = np.mean(load_pred)
    mean_true = np.mean(load_true)
    MAPE = np.mean(np.abs(load_true - load_pred) / (np.abs(load_true) + 1e-8))
    SMAPE = 2 * np.mean(np.abs(load_true - load_pred) / (np.abs(load_true) + np.abs(load_pred) + 1e-8))
    MAE = np.mean(np.abs(load_true - load_pred))
    RMSE = np.sqrt(np.mean(np.square(load_true - load_pred)))
    RRSE = np.sqrt(np.sum(np.square(load_true - load_pred))) / (
        np.sqrt(np.sum(np.square(load_true - mean_true))) + 1e-8
    )
    numerator = np.sum((load_true - mean_true) * (load_pred - mean_pred))
    denominator = (
        np.sqrt(np.sum((load_true - mean_true) ** 2)) *
        np.sqrt(np.sum((load_pred - mean_pred) ** 2))
    )
    CORR = numerator / (denominator + 1e-8)
    return MAPE, SMAPE, MAE, RMSE, RRSE, CORR, load_pred, load_true


# %% md
# ## RUN model retraining
# %%
def run_model_retraining(seq_len=seq_len,
                         target_len=target_len,
                         mse_thresh=mse_thresh):
    # 使用新的数据加载函数
    data = load_etth1_data()
    train_data = data[:int(0.8 * len(data))]
    train_data, scaler, scaler_y = normalization(train_data)
    train_x, train_y = series_to_supervise(train_data, seq_len, target_len)

    valid_x = train_x[int(0.8 * len(train_x)):]
    valid_y = train_y[int(0.8 * len(train_y)):]
    train_x = train_x[:int(0.8 * len(train_x))]
    train_y = train_y[:int(0.8 * len(train_y))]
    input_size = train_x.shape[2]

    #     hyper-parameters define
    batch_size = 256
    lr = 0.01
    number_epoch = 80
    hidden_dim = 64
    n_layers = 2
    drop_prob = 0.5
    weight_decay = 0
    mse_thresh = 0.01

    model, train_loss_list, valid_loss_list = train_model(
        train_x, train_y, valid_x, valid_y, input_size, seq_len, target_len,
        mse_thresh, hidden_dim, n_layers, number_epoch, batch_size, lr,
        drop_prob, weight_decay)

    # plot training process
    plt.plot(train_loss_list[10:], 'm', label='train_loss')
    plt.plot(valid_loss_list[10:], 'g', label='valid_loss')
    plt.grid('both')
    plt.legend()

    # test

    test_data = data[int(0.8 * len(data)):]
    test_data = scaler.transform(test_data)
    test_x, test_y = series_to_supervise(test_data, seq_len, target_len)
    MAPE, SMAPE, MAE, RMSE, RRSE, CORR, load_pred, load_true = test_model(
        model, test_x, test_y, scaler_y, seq_len, target_len, batch_size)
    return MAPE, SMAPE, MAE, RMSE, RRSE, CORR, load_pred, load_true


# %%
# random_seed_set(16)
# MAPE, SMAPE, MAE, RMSE, RRSE, CORR, load_pred, load_true = run_model_retraining()
# print('MAPE:{:.6f},SMAPE:{:.6f},MAE:{:.6f},RMSE:{:.6f},RRSE:{:.6f},CORR:{:.6f}'.format(MAPE, SMAPE, MAE, RMSE, RRSE, CORR))
# %%
# print('MAPE:{:.6f},SMAPE:{:.6f},MAE:{:.6f},RMSE:{:.6f},RRSE:{:.6f},CORR:{:.6f}'.format(MAPE, SMAPE, MAE, RMSE, RRSE, CORR))
# %% md
# ## figure plot
# %%
# plt.figure(figsize=(20, 10))
# load_pred = load_pred.reshape(-1, 24)
# load_true = load_true.reshape(-1, 24)
# plt.plot(load_pred[:240, 23], 'm')
# plt.plot(load_true[:240, 23], 'g')
# plt.ylim(ymin=0)
# plt.show()

# %% md
# ## ETTh1 Baseline Training
# %%
# 运行训练
# if __name__ == "__main__":
#     run_etth1_baseline()

# %% md
# ## Unified Training Function
# %%
def run_training(patch_size=1, save_path="./checkpoints/model_state_dict.pt",
                 results_path="./results/results.csv", use_attention=False,
                 use_seasonality=False, seed=42):
    """统一训练函数，支持baseline和patch模式"""
    import os
    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./results', exist_ok=True)

    random_seed_set(seed)
    print(f"=== 本次运行随机种子重置为: {seed} ===")

    # 设置全局Patch开关，确保切片与模型一致
    global ENABLE_PATCH, PATCH_SIZE
    ENABLE_PATCH = (patch_size > 1)
    PATCH_SIZE = patch_size

    print(f"=== 全局Patch设置: ENABLE_PATCH={ENABLE_PATCH}, PATCH_SIZE={PATCH_SIZE} ===")

    # 加载数据（使用季节性特征开关）
    data = load_etth1_data(use_seasonality=use_seasonality)
    print(f"原始数据形状: {data.shape}")
    
    # 获取目标列索引（始终为OT列）
    target_index = data.columns.get_loc('OT')
    print(f"=== 目标列索引: {target_index} (OT列) ===")

    # 按时间顺序划分数据，避免数据泄漏
    # 前80%作为train_valid_data，后20%作为test_data
    split_idx = int(len(data) * 0.8)
    train_valid_data = data[:split_idx]
    test_data = data[split_idx:]
    
    print(f"数据划分: train_valid_data={train_valid_data.shape}, test_data={test_data.shape}")

    # 只在train_valid_data上fit归一化器
    normalized_train_valid_data, scaler, scaler_y = normalization(train_valid_data)
    
    # test_data只能用train_valid_data的scaler做transform
    normalized_test_data = scaler.transform(test_data)

    # 构建监督数据集（只从train_valid_data构造）
    seq_len = 72
    target_len = 24
    train_x, train_y = series_to_supervise(normalized_train_valid_data, seq_len, target_len)
    
    # test的监督数据只从normalized_test_data构造
    test_x, test_y = series_to_supervise(normalized_test_data, seq_len, target_len)

    # 打印维度验收
    print(f"原始数据维度验收:")
    print(f"train_x.shape: {train_x.shape} (样本数, 序列长度, 特征数)")
    print(f"train_y.shape: {train_y.shape} (样本数, 预测长度, 特征数)")
    print(f"test_x.shape: {test_x.shape} (样本数, 序列长度, 特征数)")
    print(f"test_y.shape: {test_y.shape} (样本数, 预测长度, 特征数)")

    # Patch模式下打印切片后形状验收
    if ENABLE_PATCH:
        print(f"=== Patch模式形状验收 ===")
        print(f"切片后 train_x 应为: (N, {seq_len // patch_size}, {train_x.shape[2] * patch_size})")
        print(f"切片后 train_y 应为: (N, {target_len}, {train_y.shape[2]})")

    # time_series_split()只用于train_x, train_y，不要对test_x, test_y做split
    train_x_list, train_y_list, valid_x_list, valid_y_list = time_series_split(train_x, train_y)
    train_x = train_x_list[-1]
    train_y = train_y_list[-1]
    valid_x = valid_x_list[-1]
    valid_y = valid_y_list[-1]

    # 固定超参
    input_size = train_x.shape[2]  # 使用原始特征维度
    batch_size = 256
    lr = 1e-3
    number_epoch = 40
    hidden_dim = 64
    n_layers = 2

    print(f"训练超参配置:")
    print(f"  patch_size: {patch_size}")
    print(f"  batch_size: {batch_size}")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  n_layers: {n_layers}")
    print(f"  lr: {lr}")
    print(f"  seq_len: {seq_len}")
    print(f"  target_len: {target_len}")
    print(f"  number_epoch: {number_epoch}")
    print(f"  use_attention: {use_attention}")

    # 训练模型
    model, train_loss_list, valid_loss_list = train_model(
        train_x, train_y, valid_x, valid_y, input_size, seq_len, target_len,
        mse_thresh=999999,  # 设为很大的值避免提前停止
        hidden_dim=hidden_dim, n_layers=n_layers, number_epoch=number_epoch,
        batch_size=batch_size, lr=lr, drop_prob=0, weight_decay=0,
        save_path=save_path,
        patch_size=patch_size,
        use_attention=use_attention,
        target_index=target_index
    )

    # 调用test_model进行测试评估（使用前面已经构造好的test_x和test_y）
    print("=== 开始测试评估 ===")
    MAPE, SMAPE, MAE, RMSE, RRSE, CORR, load_pred, load_true = test_model(
        model, test_x, test_y, scaler_y, seq_len, target_len, batch_size,
        model_path=save_path,
        patch_size=patch_size,
        target_index=target_index
    )
    
    print(f"测试指标: MAPE={MAPE:.4f}, SMAPE={SMAPE:.4f}, MAE={MAE:.4f}, RMSE={RMSE:.4f}, RRSE={RRSE:.4f}, CORR={CORR:.4f}")
    
    # 保存指标（包含测试指标）
    import pandas as pd
    records = pd.DataFrame({
        'input_size': [input_size],
        'seq_len': [seq_len],
        'target_len': [target_len],
        'patch_size': [patch_size],
        'use_attention': [use_attention],
        'batch_size': [batch_size],
        'hidden_dim': [hidden_dim],
        'n_layers': [n_layers],
        'lr': [lr],
        'final_train_loss': [train_loss_list[-1] if train_loss_list else 0],
        'final_valid_loss': [valid_loss_list[-1] if valid_loss_list else 0],
        'MAPE': [MAPE],
        'SMAPE': [SMAPE],
        'MAE': [MAE],
        'RMSE': [RMSE],
        'RRSE': [RRSE],
        'CORR': [CORR]
    })
    records.to_csv(results_path, index=False)
    print(f"指标已保存到: {results_path}")
    print(f"模型已保存到: {save_path}")

    return model


# %% md
# ## Patch Training Function
# %%
def run_patch_training():
    """Patch训练函数，使用patch_size=4进行切片"""
    import os
    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./results', exist_ok=True)

    print("=== PATCH 训练开始 ===")

    # 加载数据
    data = load_etth1_data()
    print(f"原始数据形状: {data.shape}")

    # 数据归一化
    normalized_data, scaler, scaler_y = normalization(data)

    # 构建监督数据集
    seq_len = 72
    target_len = 24
    patch_size = 4

    # 构建原始监督数据集（不在外部进行Patch切片）
    train_x, train_y = series_to_supervise(normalized_data, seq_len, target_len)

    # 打印维度验收
    print(f"原始数据维度验收:")
    print(f"train_x.shape: {train_x.shape} (样本数, 序列长度, 特征数)")
    print(f"train_y.shape: {train_y.shape} (样本数, 预测长度, 特征数)")

    # 数据分割
    train_x_list, train_y_list, valid_x_list, valid_y_list = time_series_split(train_x, train_y)
    train_x = train_x_list[-1]
    train_y = train_y_list[-1]
    valid_x = valid_x_list[-1]
    valid_y = valid_y_list[-1]

    # 固定超参
    input_size = train_x.shape[2]  # 使用原始特征维度，切片在train_model内部进行
    batch_size = 256
    lr = 0.01
    number_epoch = 1
    hidden_dim = 64
    n_layers = 2

    print(f"Patch训练超参配置:")
    print(f"  patch_size: {patch_size}")
    print(f"  batch_size: {batch_size}")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  n_layers: {n_layers}")
    print(f"  lr: {lr}")
    print(f"  seq_len: {seq_len}")
    print(f"  target_len: {target_len}")
    print(f"  number_epoch: {number_epoch}")

    # 训练模型
    model, train_loss_list, valid_loss_list = train_model(
        train_x, train_y, valid_x, valid_y, input_size, seq_len, target_len,
        mse_thresh=999999,  # 设为很大的值避免提前停止
        hidden_dim=hidden_dim, n_layers=n_layers, number_epoch=number_epoch,
        batch_size=batch_size, lr=lr, drop_prob=0, weight_decay=0,
        save_path="./checkpoints/patch_state_dict.pt",
        patch_size=patch_size
    )

    # 保存指标
    import pandas as pd
    records = pd.DataFrame({
        'input_size': [input_size],
        'seq_len': [seq_len],
        'target_len': [target_len],
        'patch_size': [patch_size],
        'batch_size': [batch_size],
        'hidden_dim': [hidden_dim],
        'n_layers': [n_layers],
        'lr': [lr],
        'final_train_loss': [train_loss_list[-1] if train_loss_list else 0],
        'final_valid_loss': [valid_loss_list[-1] if valid_loss_list else 0]
    })
    records.to_csv('./results/patch.csv', index=False)
    print("指标已保存到: ./results/patch.csv")
    print("模型已保存到: ./checkpoints/patch_state_dict.pt")

    print("=== Patch 训练完成 ===")

    return model


# 统一入口开关
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='DARNet-V4 可复现消融实验')
    parser.add_argument('--mode', type=str, choices=['baseline', 'patch', 'attention', 'patch_attention', 'season', 'full_model', 'all'],
                        default='baseline', help='训练模式: baseline/patch/attention/patch_attention/season/full_model/all')
    parser.add_argument('--seed', type=int, default=42, help='随机种子，默认42')

    args = parser.parse_args()

    # 设置随机种子
    random_seed_set(args.seed)
    print(f"=== 随机种子已设置为: {args.seed} ===")

    if args.mode == 'baseline':
        print("=== ETTh1 BASELINE 40EP START ===")
        run_training(patch_size=1, use_attention=False, save_path="./checkpoints/baseline_state_dict.pt",
                     results_path="./results/baseline.csv", seed=args.seed)
    elif args.mode == 'patch':
        print("=== PATCH 训练开始 ===")
        run_training(patch_size=4, use_attention=False, save_path="./checkpoints/patch_state_dict.pt",
                     results_path="./results/patch.csv", seed=args.seed)
    elif args.mode == 'attention':
        print("=== ATTENTION 训练开始 ===")
        run_training(patch_size=1, use_attention=True, save_path="./checkpoints/attention_state_dict.pt",
                     results_path="./results/attention.csv", seed=args.seed)
    elif args.mode == 'patch_attention':
        print("=== PATCH + ATTENTION 训练开始 ===")
        run_training(patch_size=4, use_attention=True, save_path="./checkpoints/patch_attention_state_dict.pt",
                     results_path="./results/patch_attention.csv", seed=args.seed)
    elif args.mode == 'season':
        print("=== SEASON 训练开始 ===")
        run_training(patch_size=1, use_attention=False, use_seasonality=True, 
                     save_path="./checkpoints/season_state_dict.pt",
                     results_path="./results/season.csv", seed=args.seed)
    elif args.mode == 'full_model':
        print("=== FULL MODEL 训练开始 ===")
        run_training(patch_size=4, use_attention=True, use_seasonality=True,
                     save_path="./checkpoints/full_model_state_dict.pt",
                     results_path="./results/full_model.csv", seed=args.seed)
    elif args.mode == 'all':
        print("=== 运行所有消融实验 ===")

        print("=== ETTh1 BASELINE 40EP START ===")
        run_training(patch_size=1, use_attention=False, save_path="./checkpoints/baseline_state_dict.pt",
                     results_path="./results/baseline.csv", seed=args.seed)

        print("=== PATCH 训练开始 ===")
        run_training(patch_size=4, use_attention=False, save_path="./checkpoints/patch_state_dict.pt",
                     results_path="./results/patch.csv", seed=args.seed)

        print("=== ATTENTION 训练开始 ===")
        run_training(patch_size=1, use_attention=True, save_path="./checkpoints/attention_state_dict.pt",
                     results_path="./results/attention.csv", seed=args.seed)

        print("=== PATCH + ATTENTION 训练开始 ===")
        run_training(patch_size=4, use_attention=True, save_path="./checkpoints/patch_attention_state_dict.pt",
                     results_path="./results/patch_attention.csv", seed=args.seed)

        print("=== SEASON 训练开始 ===")
        run_training(patch_size=1, use_attention=False, use_seasonality=True, 
                     save_path="./checkpoints/season_state_dict.pt",
                     results_path="./results/season.csv", seed=args.seed)

        print("=== FULL MODEL 训练开始 ===")
        run_training(patch_size=4, use_attention=True, use_seasonality=True,
                     save_path="./checkpoints/full_model_state_dict.pt",
                     results_path="./results/full_model.csv", seed=args.seed)

    print("=== 训练完成 ===")
