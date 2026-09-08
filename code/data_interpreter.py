# %%
import torch

torch.cuda.is_available()

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
import torch
import ast
import sys
import itertools
from collections import Counter

ROOT_FOLDER = os.path.abspath(os.path.join(os.getcwd(), ".."))

# ROOT_FOLDER = "/home/yingjie_niu/Year_3/"
sys.path.append(os.path.join(ROOT_FOLDER, "NGAT", "code", "graph_building"))
from edge_generator import EdgeGenerator
from node_generator import NodeGenerator
from graph_generator import GraphGenerator
from firmgraph_dataset import FirmRelationGraph

# %%
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device, torch.cuda.device_count()

# %% [markdown]
# ### Review ACL2018 Dataset
# 01/01/2014-01/01/2016
#
# Check the tweets available period

# %% [markdown]
# #### Dataset spy

# %%
acl18_fp = os.path.join(ROOT_FOLDER, "data", "raw", "ACL2018")
acl18_price_raw = os.path.join(acl18_fp, "price", "raw")
acl18_tweet_preprocessed = os.path.join(acl18_fp, "tweet", "preprocessed")

# %%
PERIOD_START = '2014-01-02'
PERIOD_END = '2016-01-04'
# Find out the firms whose tweets data is not from 2014-01 -- 2015-12, return a list of firms
firms = []
for ticker_folder in os.listdir(acl18_tweet_preprocessed):
    if ticker_folder[0] != '.':
        tweet_list = os.listdir(acl18_tweet_preprocessed + ticker_folder)
        if sorted(tweet_list)[0][:7] != '2014-01':
            print(ticker_folder, sorted(tweet_list)[0], sorted(tweet_list)[-1])
            if ticker_folder not in firms:
                firms.append(ticker_folder)
        if sorted(tweet_list)[-1][:7] != '2015-12':
            print(ticker_folder, sorted(tweet_list)[0], sorted(tweet_list)[-1])
            if ticker_folder not in firms:
                firms.append(ticker_folder)

print(len(firms), firms)

# %%
df = pd.read_csv("/home/yingjie_niu/Year_3/GNN_longterm/data/raw/ACL2018/price/raw/AAPL.csv")
df['log return'] = 100 * np.log(df['Adj Close'] / df['Adj Close'].shift(1))


def rescale(df):
    df["log C/O"] = 100 * np.log(df['Close'] / df['Open'])
    df["log H/O"] = 100 * np.log(df['High'] / df['Open'])
    df["log L/O"] = 100 * np.log(df['Low'] / df['Open'])
    ## We normalize Open and Volumn
    cols_to_seperate_norm = ['Open', 'Volume']
    df[cols_to_seperate_norm] = df[cols_to_seperate_norm].apply(lambda x: (x - x.mean()) / x.std())
    return df


df = rescale(df)
begin_idx = df[df['Date'] == "2014-01-02"].index[0]
end_idx = df[df['Date'] == "2016-01-04"].index[0]
for m in [1, 5, 10, 21]:
    # for each step length, calculate the r_mean and std of log returns
    df['r_mean' + str(m)] = df['log return'].shift(-m).rolling(window=m).mean()
    df['std' + str(m)] = df['log return'].shift(-m).rolling(window=m).std()

    # # 计算r_mean 和 std的第二种方法
    # df['r_mean'+str(m)] = df['log return'].rolling(window=m+self.estimation_window).mean().shift(-m)
    # df['std'+str(m)] = df['log return'].rolling(window=m+self.estimation_window).std().shift(-m)

    # for each step length, also calculate the past 5-day 10-day and 21-day average of log return, these are used in the input (node features)
    df['hist_rmean' + str(m)] = df['log return'].rolling(window=m).mean()
    df['hist_std' + str(m)] = df['log return'].rolling(window=m).std()
target_df = df.iloc[begin_idx - 2 * 21 + 1:end_idx]
target_df.reset_index(drop=True, inplace=True)
target_df[['log return', 'hist_rmean1', 'r_mean1']].head(20)

# %%
target_df['log return'][5:10].mean()

# %%
day = "2014-02-04"
ticker_target_df = target_df
estimation_end = ticker_target_df.index[ticker_target_df[
                                            'Date'] == day].item() + 1  # datafrme slice need the ending index+1 so that the record of that day be included
estimation_start = estimation_end - 21
ticker_target_df_of_day = ticker_target_df.iloc[estimation_start:estimation_end]
ticker_day_Y = ticker_target_df_of_day[
    ['r_mean1', 'r_mean5', "hist_rmean5", 'r_mean10', "hist_rmean10", 'r_mean21', "hist_rmean21"]]
y = ticker_day_Y.iloc[-1]
print(f"y:{y}")
ticker_day_Y

# %%
iter = [("r_mean1", 1), ("r_mean5", 5), ("r_mean10", 10), ("r_mean21", 20)]
for col, lagging in iter:
    print(ticker_day_Y[col].iloc[-1], ticker_day_Y[col].iloc[-1 - lagging])
    y[col] = 1 if ticker_day_Y[col].iloc[-1] > ticker_day_Y[col].iloc[-1 - lagging] else 0
print(y)

# %%
aapl_p['log return'][1:6].mean()

# %%
aapl_p['r_mean5'] = aapl_p['log return'].shift(-5).rolling(window=5).mean()
aapl_p['std5'] = aapl_p['log return'].shift(-5).rolling(window=5).std()
start = aapl_p[aapl_p['Date'] == PERIOD_START].index[0]
end = aapl_p[aapl_p['Date'] == PERIOD_END].index[0]
aapl_p.iloc[start:end]


# %% [markdown]
# #### Calculate the prediction target
#
# * mean of return
# * mean of volatility
#
# Different from most stock trend/movement prediction tasks which is usually predicting next day movement (positive/negative), we are interested in a long-term average of the return and volatility. Let R(t) represent the return on day t, we want to learn a mapping function $F_1()$ and $F_2()$ that
#
# $$F_1([R(t-n, R(t)], N(t))) = mean([R(t+1),R(t+m)])$$
#
# $$F_2([R(t-n, R(t)], N(t))) = std([R(t+1),R(t+m)])$$
#
# where $n$ is the estimation window length, and $m$ is the step length. $N(t)$ represent the textual data on day $t$, usually tweets or news (depends on the dataset). Following the experiment setting in [1], they have $n = 100$ on small dataset (1-year) and $n = 250$ for large dataset (2-years). And they set $m = [1, 5, 30]$ which represents 1-day, 5-days, and 30-days ahead prediction respectively. Paper [2] also sets the step length to be 1-day, 1-week and 1-month. However, since the transcation data they used is Bitcoin which is tradable 24/7, in their case 1-week = 7-days not 5 trading days, and similarly 1-month = 30 days not 22 trading days.
#
# In our experiment setting, we try $n = [20,40, 100, 250]$, and $m = [ 5, 10, 20]$, which represents 1-week, 2-weeks, and 1-month, respectively. We omit m=1-day because we cannot calculate the realized std of 1-day based on daily return data.
#
# Since the text data (tweet) in ACL2018 dataset is available from 2014/01/01 - 2015/12/31, $t \in [2014/01/01 - 2015/12/31]$. And we need $n$ day ahead historical data as well.
#
# [1] Wu, K., Karmakar, S. A model-free approach to do long-term volatility forecasting and its variants. Financ Innov 9, 59 (2023). https://doi.org/10.1186/s40854-023-00466-6
#
# [2] Shen, D., Urquhart, A. and Wang, P., 2020. Forecasting the volatility of Bitcoin: The importance of jumps and structural breaks. European Financial Management, 26(5), pp.1294-1323.

# %%
def generate_target_for_one_stock(csv_f, estimation_window=100, dataset='acl2018'):
    """
    Input: a price dataframe of one stock, estimation window length, dataset name
    Output: a dataframe with the prediction targets as columns
    """
    steps = [5, 10, 20]
    if dataset == 'acl2018':
        # log return = 100 * r(t+1)/r(t)
        csv_f['log return'] = 100 * np.log(csv_f['Adj Close'] / csv_f['Adj Close'].shift(1))
        begin_idx = csv_f[csv_f['Date'] == PERIOD_START].index[0]
        end_idx = csv_f[csv_f['Date'] == PERIOD_END].index[0]
        # for each step length, calculate the r_mean and std of log returns
        for m in steps:
            csv_f['r_mean' + str(m)] = csv_f['log return'].shift(-m).rolling(window=m).mean()
            csv_f['std' + str(m)] = csv_f['log return'].shift(-m).rolling(window=m).std()
        if begin_idx < estimation_window:
            raise ValueError(f'Historical data length not enough. There are {begin_idx} days ahead of 2014-01-02!')
    return csv_f.iloc[begin_idx:end_idx]


# %%
df = generate_target_for_one_stock(aapl_p, estimation_window=20)

# %%
df.shape

# %%
plt.figure(figsize=(20, 5))
plt.plot(df['r_mean20'], label='20')
plt.plot(df['r_mean10'], label='10')
plt.plot(df['r_mean5'], label='5')
plt.title('Log Return Mean in 5/10/20 steps')
plt.xlabel('Trading Day')
plt.ylabel('Log Return Mean')
plt.legend()
plt.show()

# %%
plt.figure(figsize=(20, 5))
plt.plot(df['std20'], label='20')
plt.plot(df['std10'], label='10')
plt.plot(df['std5'], label='5')
plt.title('Log Return std in 5/10/20 steps')
plt.xlabel('Trading Day')
plt.ylabel('Log Return Mean')
plt.legend()
plt.show()


# %% [markdown]
# #### Build Edge based on Tweets

# %%
def read_day_tweet(tweet_fp):
    """
    Read the tweets file of a firm on a single day.
    Input: tweet file path
    Output: return a list of dicts of tweets on this day
    """
    with open(tweet_fp, 'r') as f:
        tweets = []
        for line in f:
            line_dict = json.loads(line)
            tweets.append(line_dict)
    return tweets


# %%
aapl_2014_1_1_tweet = read_day_tweet(tweet_fp=acl18_tweet_preprocessed + 'AAPL/2014-01-01')
aapl_2014_1_1_tweet

# %%
for tweet in aapl_2014_1_1_tweet:
    tweet_arr = np.array(tweet['text'])
    # find the index of "$" in the tweet
    idx = tweet_arr == '$'
    # find the index of any word following "$", which is usually the ticker of a firm
    idx = np.array(pd.Series(idx).shift(1)).astype('bool')
    idx[0] = False
    print('Tweet #', tweet['user_id_str'], tweet_arr[idx])
    for firm in tweet_arr[idx]:
        if firm.upper() in VALID_NODE_LIST:
            print(firm)

# %% [markdown]
# #### Test Graph Generators

# %%
import yfinance

raw_data = yfinance.download(tickers="^GSPC", start="2022-05-31",
                             end="2024-05-21", interval="1d")

# %%
raw_data

# %%
dataset = FirmRelationGraph(
    dynamic="cooccur",
    num_node_features=8,
    matrix_node_features=True,
    node_normalize='mean',
    num_Y_features=4,
    Y_type="mean",
    classification=True,
    graph_level_task=False,
    estimation_window=21,
    num_edge_features=1,
    weighted_edge=True,
    threshold=1,
    root=os.path.join(ROOT_FOLDER, "GNN_longterm", "data", "graph_sets"),
    name="ACL2018"

)

# %%
dataset[3]

# %%
dataset[3].edge_index, dataset[3].edge_attr

# %%
dataset.graphgenerator.index_to_ticker(60)

# %%
dev_len, test_len = int(len(dataset) / 10), int(len(dataset) / 10)
train_len = len(dataset) - dev_len - test_len
dataset[0], len(dataset), train_len + dev_len, test_len

# %%
last_period = dataset.graphgenerator.last_period
keys = list(last_period.keys())
last_period[keys[0]]

# %%
dataset.graphgenerator.node_generator.graph_target_df.iloc[41:][
    ["Adj Close", "log return", 'r_mean1', 'hist_rmean1', 'r_mean5', 'hist_rmean5', 'r_mean10', 'hist_rmean10',
     'r_mean21', 'hist_rmean21']]

# %%
input = torch.tensor([0.6, 0.4])
torch.where(input < 0.5, 0, 1)

# %%
dataset.graphgenerator.node_generator.valid_trading_days[train_len + dev_len:]

# %%
for i in range(len(dataset)):
    if torch.nan in dataset[i].y:
        print(i)

# %%
a = torch.rand(5)
a.le(0.5) == torch.tensor([0, 0, 0, 1, 1])

# %%
a.le(0.5).long(), a

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Generate synthetic data
torch.manual_seed(0)
X = torch.randn(1000, 20)  # 1000 samples, 20 features each
y = (torch.rand(1000) > 0.5).float()  # Binary labels

# Create a DataLoader
dataset = TensorDataset(X, y)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)


# Define the model
class BinaryClassificationModel(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassificationModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 10)
        self.layer2 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x


# Initialize the model, loss function, and optimizer
model = BinaryClassificationModel(input_dim=20)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in data_loader:
        # Forward pass
        outputs = model(inputs)
        print(outputs.shape, labels.shape)
        print(outputs[:5], labels[:5])
        loss = criterion(outputs, labels.unsqueeze(1))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation
with torch.no_grad():
    model.eval()
    outputs = model(X)
    predicted = (outputs.squeeze() > 0.5).float()
    accuracy = (predicted == y).sum().item() / y.size(0)
    print(f'Accuracy: {accuracy * 100:.2f}%')

# %%
q = []
q.append(1)
q.append(2)
q.append(3)
q.pop(0)
q.append(4)
q

# %%
dev_len, test_len = int(len(dataset) / 10), int(len(dataset) / 10)
train_len = len(dataset) - dev_len - test_len
train_len, dev_len, test_len

# %%
edge_of_each_day = dataset.graphgenerator.edge_generator.edge_of_each_day
for day in edge_of_each_day:
    values = edge_of_each_day[day].values()
    try:
        print(day, ":", max(values))
    except ValueError:
        pass

# %%
for pair in edge_of_each_day['2014-01-01']:
    if pair[0] == 'goog':
        print(pair)

# %%
edge_of_each_day['2014-10-04']

# %%
from torch_geometric.utils import degree

for i in range(len(dataset)):
    data = dataset[i]
    num_degree = degree(data.edge_index[0], num_nodes=data.x.shape[0])
    print(min(num_degree), max(num_degree))

# %%
len(dataset), len(os.listdir("/home/yingjie_niu/Year_3/GNN_longterm/data/raw/SPNews/news/preprocessed_daily_2"))

# %%
import itertools

permutations = list(itertools.permutations(["aapl", "aapl", "aapl", "aapl"], 2))
permutations, len(permutations), 17 * 16

# %%
edge_of_each_day["2023-05-06"]

# %%
for pair in edge_of_each_day["2023-05-06"]:
    firm1, firm2 = pair
    if edge_of_each_day["2023-05-06"][pair] == 10:
        if firm1 != firm2:
            print(pair)

# %%
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

# %%
dataset, dataset[1], dataset[1].edge_index

# %%
len(dataset.graphgenerator.edge_generator.edge_of_each_day['2014-01-02'])

# %%
for i in range(len(dataset)):
    a = dataset[i].edge_index[0] == dataset[i].edge_index[1]
    if a.sum() != len(dataset.graphgenerator.valid_firm_list):
        print(i)


# %%
def weighted_sum_dicts(dict_list):
    # Calculate the weights for each dictionary
    n = len(dict_list)
    weights = [(i + 1) / sum(range(1, n + 1)) for i in range(n)]

    # Initialize an empty dictionary to store the weighted sums
    weighted_sum = {}

    # Iterate through the list of dictionaries
    for idx, d in enumerate(dict_list):
        weight = weights[idx]
        for key, value in d.items():
            if key in weighted_sum:
                weighted_sum[key] += value * weight
            else:
                weighted_sum[key] = value * weight

    return weighted_sum


# %%
day = '2014-01-09'


def get_prev_days(day: str, prev_len: int = 5, type: str = "natual"):
    day_index = dataset.graphgenerator.valid_trading_days.index(day)
    if day_index >= prev_len - 1:  # if there are more than prev_len days before the "day"
        prev_trading_days = dataset.graphgenerator.valid_trading_days[day_index - prev_len + 1: day_index + 1]
    else:
        prev_trading_days = dataset.graphgenerator.valid_trading_days[: day_index + 1]
    for i in range(len(prev_trading_days)):
        try:
            first_trading_day = dataset.graphgenerator.valid_natural_days.index(prev_trading_days[i])
            break
        except ValueError:
            pass
    for i in reversed(range(len(prev_trading_days))):
        try:
            last_trading_day = dataset.graphgenerator.valid_natural_days.index(prev_trading_days[-1])
            break
        except ValueError:
            pass
    prev_natual_days = dataset.graphgenerator.valid_natural_days[first_trading_day:last_trading_day + 1]
    if type == "natual":
        return prev_natual_days
    elif type == "trading":
        return prev_trading_days
    else:
        raise ValueError(f"Wrong Argument Value for 'type'={type}")


get_prev_days(day, prev_len=7)

# %%
valid_dynamic_method = ['static', 'cooccur', 'cooccurNcorr', 'corr_rmean5', 'corr_rmean10', 'corr_rmean20', 'corr_std5',
                        'corr_std5', 'corr_std5', ]
valid_dynamic_method[3:]

# %%
prev_edge_of_days = []
for i in range(len(prev_natual_days)):
    prev_edge_of_days.append(dataset.graphgenerator.edge_generator.edge_of_each_day[prev_natual_days[i]])
prev_edge_of_days

# %%
result = weighted_sum_dicts(prev_edge_of_days)
result

# %%
len(result)

# %%
unic_keys = []
for dict in prev_edge_of_days:
    keys = list(dict.keys())
    unic_keys.extend(keys)
len(set(unic_keys))

# %%
dataset.graphgenerator.edge_generator.tweets_of_each_day

# %%
dataset.graphgenerator.edge_generator.edge_of_each_day['2014-01-20']

# %%
print(dataset.graphgenerator.valid_firm_list)

# %%
goog = dataset.graphgenerator.node_generator.ticker_target_dic['GOOG']
aapl = dataset.graphgenerator.node_generator.ticker_target_dic['GOOG']
# dataset.graphgenerator.edge_generator.edge_of_each_day
df = aapl

# %%
# flash_crash = 9
# dev_len, test_len = int(len(dataset)/10), int(len(dataset)/10*1.5)-flash_crash
# train_len = len(dataset)-dev_len-test_len-flash_crash
# print(train_len, dev_len, test_len)

## Train, Dev, Test split
dev_len, test_len = int(len(dataset) / 10), int(len(dataset) / 10)
train_len = len(dataset) - dev_len - test_len
print(train_len, dev_len, test_len)

# %%
dataset.graphgenerator.edge_generator.edge_of_each_day['2014-01-02']

# %%
for key in dataset.graphgenerator.edge_generator.edge_of_each_day['2014-01-02']:
    if key[0] == key[1]:
        print(key)

# %% [markdown]
# #### Check the distribution of Train/Val/Test set

# %%
import seaborn as sns

feat = 'std5'
train_feat = df.iloc[:train_len][feat]
dev_feat = df.iloc[train_len: train_len + dev_len][feat]
test_feat = df.iloc[train_len + dev_len: train_len + dev_len + test_len][feat]
sns.kdeplot(train_feat, shade=True, color='r', label='train')
sns.kdeplot(dev_feat, shade=True, color='g', label='dev')
sns.kdeplot(test_feat, shade=True, color='b', label='test')
plt.xlabel('Feature')
plt.legend()
plt.show()

# %%
feat = 'std20'
df.iloc[:train_len][feat].plot()
df.iloc[train_len: train_len + dev_len][feat].plot()
df.iloc[train_len + dev_len: train_len + dev_len + test_len][feat].plot()
# df.iloc[train_len+dev_len : train_len+dev_len+test_len][feat].plot()

# %%
print("Start: ", train_len + 26, "End: ", train_len + 34)
df.iloc[train_len + 20:train_len + 40]

# %%
df.iloc[:train_len].r_mean10.plot()
df.iloc[train_len: train_len + dev_len].r_mean10.plot()
df.iloc[train_len + dev_len: train_len + dev_len + test_len].r_mean10.plot()

# %%
df.iloc[:train_len].r_mean20.plot()
df.iloc[train_len: train_len + dev_len].r_mean20.plot()
df.iloc[train_len + dev_len: train_len + dev_len + test_len].r_mean20.plot()

# %% [markdown]
# #### Try Something

# %%
nodegenerator = NodeGenerator(
    dataset_name="SPNews",
    num_node_features=8,
    num_Y_features=4,
    Y_type="mean",
    estimation_window=21,
    matrix_node_features=True,
    classification=True,
    node_normalize="mean"
)

# %%
X, Y = [], []
for day in nodegenerator.valid_trading_days:
    x = np.array(nodegenerator.build_graph_level_X(day=day))
    y, _ = np.array(nodegenerator.build_graph_level_y(day=day))
    X.append(x)
    Y.append(y)

# %%
from torch.utils.data import Dataset, DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        x = x.to(device=device)
        y = y.to(device=device)
        return x, y


# %%
dataset = CustomDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Example Usage
for batch_idx, (data, target) in enumerate(dataloader):
    print(f"Batch {batch_idx + 1}:")
    print(f"Data: {data.shape}")
    print(f"Target: {target.shape}")
    print()

# %%
from torch.utils.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ROOT_FOLDER = "/home/yingjie/Year_3/"


graph_params = {
    ##  -------- Graph Params ---------
    "dynamic": "cooccur",
    ##  -------- Node Params ---------
    "num_node_features": 8,
    "matrix_node_features": True,
    "node_normalize": 'mean',
    "num_Y_features": 3,
    "Y_type": "std",
    "classification": False,
    "graph_level_task": True,
    "estimation_window": 21,
    ## -------- Edge Params ----------
    "num_edge_features": 1,
    "weighted_edge": True,
    "threshold": 0,
    ## -------- Dataset Params --------
    "root": os.path.join(ROOT_FOLDER, "GNN_longterm", "data", "graph_sets"),
    "name": "ACL2018"
}

nodegenerator = NodeGenerator(
    dataset_name=graph_params['name'],
    num_node_features=graph_params['num_node_features'],
    num_Y_features=graph_params['num_Y_features'],
    Y_type=graph_params['Y_type'],
    estimation_window=graph_params['estimation_window'],
    matrix_node_features=graph_params['matrix_node_features'],
    classification=graph_params['classification'],
    node_normalize=graph_params['node_normalize']
)


class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float64)
        y = torch.tensor(self.labels[idx], dtype=torch.float64)
        x = x.to(device=device)
        y = y.to(device=device)
        return x, y


X, Y = [], []
for day in nodegenerator.valid_trading_days:
    x = np.array(nodegenerator.build_graph_level_X(day=day))
    y, _ = nodegenerator.build_graph_level_y(day=day)
    y = np.array(y)
    X.append(x)
    Y.append(y)
dataset = CustomDataset(X, Y)

# %%
x, y = dataset[0]
x.shape, y.shape, y

# %% [markdown]
# Example Values:
# Assuming:
#
# source = [0, 2, 4]
# torch.arange(len(source)) = [0, 1, 2]
# When a_input[:, source, torch.arange(len(source)), :] is used, it will update the tensor as follows:
#
# For the first head (head 0):
# * a_input[0, 0, 0, :] is updated with concatenated_features[0, 0, :]
# * a_input[0, 2, 1, :] is updated with concatenated_features[0, 1, :]
# * a_input[0, 4, 2, :] is updated with concatenated_features[0, 2, :]
#
# For the second head (head 1):
# * a_input[1, 0, 0, :] is updated with concatenated_features[1, 0, :]
# * a_input[1, 2, 1, :] is updated with concatenated_features[1, 1, :]
# * a_input[1, 4, 2, :] is updated with concatenated_features[1, 2, :]
#

# %%
x = torch.FloatTensor(dataset[0].x.shape[0], dataset[0].x.shape[2])
y = torch.FloatTensor(dataset[0].y.shape[0], dataset[0].y.shape[1])
print(x.shape, y.shape)
source, target = dataset[0].edge_index
weight = torch.FloatTensor(8, 76, dataset[0].x.shape[2], dataset[0].y.shape[1])
h = torch.matmul(x.reshape(-1, 1, dataset[0].x.shape[2]), weight).reshape(8, 76, -1)
# a_input = torch.cat([h[:,source], h[:,target]], dim=2)
source_feat = h[:, source, :]
target_feat = h[:, target, :]
concatenated_features = torch.cat([source_feat, target_feat], dim=-1)  # Shape: (8, num_edges, 2 * y.shape[1])

a_input = torch.zeros(size=(8, 76, len(source), 2 * h.shape[2]))
a_input[:, source, torch.arange(len(source)), :] = concatenated_features
# for i in range(len(source)):
#     s = source[i]
#     t = target[i]
#     s_feat = h[:, s, :]
#     t_feat = h[:, t, :]
#     concated_feat = torch.cat((s_feat, t_feat), dim=-1)
#     a_input[:, s, i, :] = concated_feat
a = nn.Parameter(torch.ones(size=(8, 76, 2 * dataset[0].y.shape[1], 1)))
e = F.leaky_relu(torch.sum(torch.matmul(a_input, a), dim=1), negative_slope=0.2)
N = h.shape[1]
attention = -1e20 * torch.ones([8, N, N], requires_grad=True)
attention[:, source, target] = e[:, :, 0]
attention = F.softmax(attention, dim=2)
attention = F.dropout(attention, 0.2)
h_prime = torch.matmul(attention, h)
print(h_prime.shape)
concat = False
if concat:
    h_prime = h_prime.permute(1, 0, 2).reshape(N, -1)
else:
    h_prime = torch.sum(h_prime, dim=0)
h_prime.shape

# %%
torch.ones([92]).reshape(-1)

# %%
attention[:, source, target] = e[:, :, 0]

# %%
source, target

# %%
h.shape, e.shape, torch.arange(len(source))

# %%
a_input = torch.zeros(size=(8, 76, len(source), 2 * h.shape[2]))
for i in range(len(source)):
    s = source[i]
    t = target[i]
    s_feat = h[:, s, :]
    t_feat = h[:, t, :]
    concated_feat = torch.cat((s_feat, t_feat), dim=-1)
    a_input[:, s, i, :] = concated_feat

# %%
a.shape, a_input.shape,
torch.sum(torch.matmul(a_input, a), dim=1).shape

# %%
8 * 76 * 6 * 1, 8 * 76 * 92 * 6, a_input[;, ]

# %%


# %% [markdown]
# ### Review SPNews Dataset

# %%

spnews_fp = "/home/yingjie_niu/Year_3/GNN_longterm/data/raw/SPNews/"
spnews_price_raw = spnews_fp + "price/"
spnews_news_raw = spnews_fp + "news/raw/"
spnews_news_raw_202310 = spnews_fp + "news/raw_202310_"

# %%
day_news = pd.read_csv("/home/yingjie_niu/Year_3/GNN_longterm/data/raw/SPNews/news/preprocessed_daily_2/2024-04-03.csv")
day_news['related_companies'] = day_news.apply(
    lambda x: ast.literal_eval(x['Related Company']) if x['Related Company'][0] == "[" else [x['Company']], axis=1)
day_news = day_news.drop_duplicates(subset='Link', keep="first")
day_news.drop(columns=['Publisher', 'Link', 'ProviderPublishTime', 'Type'], inplace=True)
# df_exploded = day_news.explode('related_companies')
# mask = df_exploded['Company'] != df_exploded['related_companies']
day_news[day_news['Text'].str.startswith("[In this article")]

# %%
1 + 2 + 3 + 4 + 5

# %%
n = 2
weights = [(i + 1) / sum(range(1, n + 1)) for i in range(n)]
weights

# %%
eg = EdgeGenerator(dataset_name="SPNews", num_edge_features=1, weighted_edge=True, threshold=0, dynamic=True)
day_news = eg.read_day_news(day='2022-09-24')
day_news['related_companies'] = day_news.apply(
    lambda x: ast.literal_eval(x['Related Company']) if x['Related Company'][0] == "[" else [x['Company']], axis=1)
# df_exploded = day_news.explode('related_companies')
# df_exploded['related_companies']
day_news['permutations'] = day_news.apply(
    lambda x: list(itertools.permutations(x['related_companies'], 2)) if len(x['related_companies']) > 1 else [
        (x['related_companies'][0], x['related_companies'][0])], axis=1)
all_pairs = [pair for sublist in day_news["permutations"] for pair in sublist]
pair_counts = Counter(all_pairs)
# day_edges_dict = {pair: count for pair, count in pair_counts.items() if pair[0] in eg.valid_firm_list and pair[1] in eg.valid_firm_list}
# print(len(day_edges_dict))
# day_edges_dict = eg.add_self_edge(day_edges_dict)
# print(len(day_edges_dict))
day_news

# %%
day_news['related_companies'][1]) == 1

# %%
firm_count = {}

for pair, count in pair_counts.items():
    firm1, firm2 = pair
if firm1 in firm_count:
    firm_count[firm1] += 1
else:
    firm_count[firm1] = 1

if firm2 in firm_count:
    firm_count[firm2] += 1
else:
    firm_count[firm2] = 1
firm_count

# %%
firm_pair = []

for pair, count in pair_counts.items():
    firm1, firm2 = pair
if "AAPL" == firm1:
    firm_pair.append(pair)

firm_pair

# %%
for pair, count in pair_counts.items():
    firm1, firm2 = pair
print(firm1)
break

# %%
day_news = eg.read_day_news(day='2022-09-24')
day_news['related_companies'] = day_news.apply(
    lambda x: ast.literal_eval(x['Related Company']) if x['Related Company'][0] == "[" else [x['Company']], axis=1)
day_edges_dict = {}
df_exploded = day_news.explode('related_companies')
mask = df_exploded['Company'] != df_exploded['related_companies']
df_exploded = df_exploded[mask]
# df_exploded

# Generate pairs (source, related) and (related, source)
pairs = pd.concat([
    df_exploded[['Company', 'related_companies']],
    df_exploded[['related_companies', 'Company']].rename(
        columns={'related_companies': 'Company', 'Company': 'related_companies'})
])
# Filter out the ticker that is not in valid_firm_list
pairs.reset_index(drop=True, inplace=True)
pairs = pairs[pairs['related_companies'].isin(eg.valid_firm_list)]
pairs.reset_index(drop=True, inplace=True)
pairs = pairs[pairs['Company'].isin(eg.valid_firm_list)]
pairs.reset_index(drop=True, inplace=True)

# Count occurrences of each pair
pair_counts = pairs.groupby(['Company', 'related_companies']).size().reset_index(name='count')

# Set the index to be a MultiIndex with source and related_companies
pair_counts = pair_counts.set_index(['Company', 'related_companies'])

# Convert to dictionary
day_edges_dict = pair_counts['count'].to_dict()
print(len(day_edges_dict))
day_edges_dict = eg.add_self_edge(day_edges_dict)
print(len(day_edges_dict))

# %%
df = pd.read_csv("/home/yingjie_niu/Year_3/GNN_longterm/data/raw/SPNews/price/AAL.csv")
df['log return'] = 100 * np.log(df['Close'] / df['Close'].shift(1))  # Calculate Log return = 100 * r(t+1)/r(t)
for m in [1, 5, 10, 21]:
    # for each step length, calculate the r_mean and std of log returns
    df['r_mean' + str(m)] = df['log return'].shift(-m).rolling(window=m).mean()
    df['std' + str(m)] = df['log return'].shift(-m).rolling(window=m).std()

# %%
import seaborn as sns

## Train, Dev, Test split
dev_len, test_len = int(399 / 10), int(399 / 10)
train_len = 399 - dev_len - test_len
print(train_len, dev_len, test_len)
feat = 'std10'
train_feat = df.iloc[:train_len][feat]
dev_feat = df.iloc[train_len: train_len + dev_len][feat]
test_feat = df.iloc[train_len + dev_len: train_len + dev_len + test_len][feat]
sns.kdeplot(train_feat, shade=True, color='r', label='train')
sns.kdeplot(dev_feat, shade=True, color='g', label='dev')
sns.kdeplot(test_feat, shade=True, color='b', label='test')
plt.xlabel('Feature')
plt.legend()
plt.show()

# %%
# List of valid firms
valid_firms = {'HUM', 'CNC', 'ITW', 'BA'}

# Flatten the Series into a list of tuples
all_pairs = [pair for sublist in day_news["permutations"] for pair in sublist]

# Count the occurrences of each tuple using Counter
pair_counts = Counter(all_pairs)

# Convert the Counter to a dictionary (if needed)
filtered_pair_counts = {pair: count for pair, count in pair_counts.items() if
                        pair[0] in valid_firms and pair[1] in valid_firms}

filtered_pair_counts

# %%
# ## Generate preprocessed_tickerwise


# # Specify the folder containing the CSV files
# save_path = '/home/yingjie_niu/Year_3/GNN_longterm/data/raw/SPNews/news/preprocessed_tickerwise'

# # Define the header
# header = ['Id', 'Company', 'Date', 'Title', 'Publisher', 'Link', 'ProviderPublishTime', 'Type', 'Related Company', 'Text']


# # Function to convert date format to string
# def convert_date_format(date_str):
#     # print(date_str)
#     if len(date_str.split("-")) == 3:
#         y, m, d = date_str.split("-")[0], date_str.split("-")[1], date_str.split("-")[2]
#         # print(y,m,d,len(m), len(d))
#         if len(m) == 1:
#             m = "0"+m
#         if len(d) == 1:
#             d = "0"+d
#         # print(y+'-'+m+'-'+d)
#         return y+'-'+m+'-'+d
#     else:
#         return date_str


# # Iterate through each file in the folder
# for filename in os.listdir(spnews_news_raw_202310):
#     # Check if the file is a CSV file
#     if filename.endswith('.csv'):
#         file_path = os.path.join(spnews_news_raw, filename)
#         file_path2 = os.path.join(spnews_news_raw_202310, filename)

#         # Read the CSV file
#         df = pd.read_csv(file_path, header=None)
#         df2 = pd.read_csv(file_path2, header=None)

#         # Add the header
#         df.columns = header
#         df2.columns = header

#         # Concat df and df2
#         df_concat = pd.concat([df, df2])
#         df_concat.reset_index(drop=True, inplace=True)

#         # Convert the 'Date' column format to string
#         df_concat['Date'] = df_concat['Date'].apply(lambda x: convert_date_format(x) if pd.notnull(x) else x)

#         # Save the CSV file back
#         df_concat.to_csv(os.path.join(save_path, filename), index=False)

#         print(f'Updated {filename} with header.')


# %%
df = pd.read_csv("/home/yingjie_niu/Year_3/GNN_longterm/data/raw/SPNews/news/preprocessed_daily_2/2022-09-21.csv")
df.drop_duplicates(subset="Link", keep="first", inplace=True)
df

# %%
# # Generate preprocessed_daily


node_generator = NodeGenerator(
    "SPNews",
    num_node_features=8,
    node_normalize="mean",
    num_Y_features=3,
    classification=False,
    Y_type="std",
    estimation_window=21,
    matrix_node_features=True
)

# for day in tqdm(node_generator.valid_natural_days):
#     if day+".csv" in os.listdir("/home/yingjie_niu/Year_3/GNN_longterm/data/raw/SPNews/news/preprocessed_daily_2"):
#         pass
#     else:
#         dfs = []
#         for ticker in node_generator.valid_firm_list:
#             df = pd.read_csv("/home/yingjie_niu/Year_3/GNN_longterm/data/raw/SPNews/news/preprocessed_tickerwise/"+ticker+'_news.csv')
#             df.drop_duplicates(subset="Link", keep="first", inplace=True)
#             df = df[df['Date'] == day]
#             dfs.append(df)
#         day_news = pd.concat(dfs, ignore_index=True)
#         day_news.to_csv("/home/yingjie_niu/Year_3/GNN_longterm/data/raw/SPNews/news/preprocessed_daily_2/"+day+'.csv')

# %%
len(node_generator.valid_firm_list), len(node_generator.all_firm_list)

# %%
import itertools
import ast

day_edges_dict = {}
# for i in range(len(df['Related Company'])):
#     source = df['Company'].iloc[i]
#     target_list = df['Related Company'].iloc[i]
#     if target_list[0] == "[":
#         target_list = ast.literal_eval(target_list)
#         target_list.append(source)
#         permutations = list(itertools.permutations(target_list , 2))
#         for firm_pair in permutations:
#             if firm_pair not in day_edges_dict:
#                 day_edges_dict[firm_pair] = 1
#             else:
#                 day_edges_dict[firm_pair] += 1
df['related_companies'] = df.apply(
    lambda x: ast.literal_eval(x['Related Company']) if x['Related Company'][0] == "[" else [x['Company']], axis=1)
# Explode the related_companies list into multiple rows
df_exploded = df.explode('related_companies')
# df_exploded
# Generate pairs (source, related) and (related, source)
pairs = pd.concat([
    df_exploded[['Company', 'related_companies']],
    df_exploded[['related_companies', 'Company']].rename(
        columns={'related_companies': 'Company', 'Company': 'related_companies'})
])
print(pairs.shape)
pairs[pairs['related_companies'].isin(node_generator.valid_firm_list)]
# # Count occurrences of each pair
# pair_counts = pairs.groupby(['Company', 'related_companies']).size().reset_index(name='count')
# pair_counts


# %%
# Set the index to be a MultiIndex with source and related_companies
pair_counts = pair_counts.set_index(['Company', 'related_companies'])

# Convert to dictionary
pairs_dict = pair_counts['count'].to_dict()
pairs_dict
# print(day_edges_dict)

# %%
dataset = FirmRelationGraph(
    dynamic="cooccur",
    num_node_features=8,
    matrix_node_features=True,
    node_normalize='mean',
    num_Y_features=3,
    Y_type="std",
    classification=False,
    estimation_window=21,
    num_edge_features=1,
    graph_level_task=False,
    weighted_edge=True,
    threshold=0,
    root="/home/yingjie/Year_3/GNN_longterm/data/graph_sets",
    name="SPNews")

# %%
from torch_geometric.data import DataLoader

dev_len, test_len = int(len(dataset) / 10), int(len(dataset) / 10)
train_len = len(dataset) - dev_len - test_len
print(test_len)
test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
print(len(test_loader))
for i in range(test_len):
    batch = dataset[i].to(device)
    print(batch)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
all_firm_list = dataset.graphgenerator.all_firm_list

valid_firm_list = dataset.graphgenerator.valid_firm_list

# Convert valid_firm_list to a set for efficient lookup
valid_firm_set = set(valid_firm_list)

# Create a mask where True represents nodes of interest
mask = torch.tensor([firm in valid_firm_set for firm in all_firm_list], dtype=torch.bool)

dataset[0]

# %%
alist = dataset.graphgenerator.edge_generator.tweets_of_each_day['2014-01-24']
seen = set()
unique_dicts = []
repeat_dicts = []
for d in alist:
    dict_str = json.dumps(d, sort_keys=True)
    if dict_str not in seen:
        seen.add(dict_str)
        unique_dicts.append(d)
    else:
        repeat_dicts.append(d)
print(len(seen))


# %%
def remove_duplicates(dict_list: list) -> list:
    """
    This function is to remove duplicate tweets in a day. It is specifically for ACL2018 dataset.
    """
    seen = set()
    unique_dicts = []

    for d in dict_list:
        # Convert the dictionary to a JSON string
        dict_str = json.dumps(d, sort_keys=True)

        if dict_str not in seen:
            seen.add(dict_str)
            unique_dicts.append(d)

    return unique_dicts


remove_duplicates(alist)

# %%
# degree = []
# for i in range(len(dataset)):
#     degree.append(dataset[i].edge_index.shape[1]/ 76)
# np.mean(degree)

degree = []
for i in range(len(dataset)):
    # print(dataset[i].edge_index.shape)
    degree.append(dataset[i].edge_index.shape[1] / 268)
np.mean(degree)

# %%
# Define the dimension N of the identity matrix
N = 4  # You can change this to any desired value

# Create a single identity matrix of shape [N, N]
identity_matrix = torch.eye(N)

# Repeat this identity matrix 3 times along a new dimension to get a tensor of shape [3, N, N]
identity_matrices = identity_matrix.unsqueeze(0).repeat(3, 1, 1)

print(identity_matrices)
print(identity_matrices.shape)

# %%
identity_matrices + identity_matrices

# %%
num_nodes = 10
edge_idx = np.ones([2, num_nodes], dtype=int)
edge_idx[:, range(num_nodes)] = range(num_nodes)  # edge_index only have self loops, i.e. [[0,0],[1,1],[2,2]...]
edge_idx_T = edge_idx.T
edge_attr = np.ones([num_nodes, 1], dtype=int)
edge_idx_T.dtype, edge_attr.dtype

# %%
ticker_target_dict = dataset.graphgenerator.node_generator.ticker_target_dic
for t in ticker_target_dict:
    # if np.any(np.isnan(ticker_target_dict[t])):
    #     print(t)
    if ticker_target_dict[t].isnull().any().any():
        print(t)

# %%
df = pd.read_csv("/home/yingjie_niu/Year_3/GNN_longterm/data/raw/SPNews/price/AAL.csv")
df[df[df['Date'] == '2022-09-20'].index[0]:df[df['Date'] == '2024-04-24'].index[0]]

# %%
dev_len, test_len = int(len(dataset) / 10), int(len(dataset) / 10)
train_len = len(dataset) - dev_len - test_len
print(train_len, dev_len, test_len)

# %%
edge_of_each_day = dataset.graphgenerator.edge_generator.edge_of_each_day
# len(edge_of_each_day.keys()), len(dataset.graphgenerator.valid_natural_days)
edge_of_each_day['2023-10-20']

# %%
df = pd.read_csv("/home/yingjie_niu/Year_3/GNN_longterm/data/raw/SPNews/price/AAL.csv")
start = df[df["Date"] == "2023-10-20"].index[0]
end = df[df["Date"] == "2023-11-20"].index[0]
print(list(df[start:end]['Date']))

# %%
source = torch.tensor([1, 2])
target = torch.tensor([2, 1])
adj = torch.zeros(3, 3)
adj[source, target] = torch.ones(len(source))
adj

# %%
day = "2014-01-02"
ticker_target_df = dataset.graphgenerator.node_generator.ticker_target_dic['AAPL']
estimation_end = ticker_target_df.index[ticker_target_df[
                                            'Date'] == day].item() + 1  # datafrme slice need the ending index+1 so that the record of that day be included
estimation_start = estimation_end - 20
ticker_target_df_of_day = ticker_target_df.iloc[
                          estimation_start:estimation_end]  # shape [estimation_window, 14], from "day-estimation_window" to "day", columns:  ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'log return', 'r_mean5', 'std5', 'r_mean10', 'std10', 'r_mean20', 'std20']
ticker_day_Y = ticker_target_df_of_day[['std5']]
y = ticker_day_Y.iloc[-1].item()
y, ticker_day_Y

# %%
a = dataset[0].x
a = a.to(device='cuda:0')
a.device

# %%
import pickle

path = '/home/yingjie_niu/Year_3/AD_GAT/data/ACL2018/'
with open(path + 'x_textual.pkl', 'rb') as handle:
    y_load = pickle.load(handle)  # 730, 198
y_load.shape

# %%


# %%
day = "2014-06-02"
ticker_target_df = dataset.graphgenerator.node_generator.ticker_target_dic['AAPL']
estimation_end = ticker_target_df.index[ticker_target_df[
                                            'Date'] == day].item() + 1  # datafrme slice need the ending index+1 so that the record of that day be included
estimation_start = estimation_end - 20
ticker_target_df_of_day = ticker_target_df.iloc[estimation_start:estimation_end]

# %%
ticker_day_Y = ticker_target_df_of_day[['r_mean5', 'r_mean10', 'r_mean20', ]]
ticker_day_Y

# %%
y = ticker_day_Y.loc[estimation_end - 1]


def to_classification_target(y, ticker_day_Y, Y_type):
    if Y_type == "mean":
        iter = [("r_mean5", 5), ("r_mean10", 10), ("r_mean20", 19)]
    elif Y_type == "std":
        iter = [("std5", 5), ("std10", 10), ("std20", 19)]
    elif Y_type == "all":
        iter = [("r_mean5", 5), ("r_mean10", 10), ("r_mean20", 19), ("std5", 5), ("std10", 10), ("std20", 19)]
    elif Y_type == "r_mean5":
        iter = [("r_mean5", 5)]
    elif Y_type == "r_mean10":
        iter = [("r_mean10", 10)]
    elif Y_type == "r_mean20":
        iter = [("r_mean20", 20)]
    elif Y_type == "std5":
        iter = [("std5", 5)]
    elif Y_type == "std10":
        iter = [("std10", 10)]
    elif Y_type == "std20":
        iter = [("std20", 19)]

    for col, lagging in iter:
        y[col] = 1 if ticker_day_Y[col].iloc[-1] > ticker_day_Y[col].iloc[-1 - lagging] else 0
    return y


post_y = to_classification_target(y, ticker_day_Y=ticker_day_Y, Y_type="mean")
print(y, post_y)

# %% [markdown]
# #### Check the distribution of Train/Val/Test set

# %%
import seaborn as sns

goog = dataset.graphgenerator.node_generator.ticker_target_dic['GOOG']
aapl = dataset.graphgenerator.node_generator.ticker_target_dic['AAPL']
# dataset.graphgenerator.edge_generator.edge_of_each_day
df = aapl

dev_len, test_len = int(len(dataset) / 10), int(len(dataset) / 10)
train_len = len(dataset) - dev_len - test_len
print(train_len, dev_len, test_len)

feat = 'std5'
train_feat = df.iloc[:train_len][feat]
dev_feat = df.iloc[train_len: train_len + dev_len][feat]
test_feat = df.iloc[train_len + dev_len: train_len + dev_len + test_len][feat]
sns.kdeplot(train_feat, shade=True, color='r', label='train')
sns.kdeplot(dev_feat, shade=True, color='g', label='dev')
sns.kdeplot(test_feat, shade=True, color='b', label='test')
plt.xlabel('Feature')
plt.legend()
plt.show()

# %%
train_feat.plot()
dev_feat.plot()
test_feat.plot()

# %% [markdown]
# ### Load Model

# %%
import sys

# ROOT_FOLDER = "/home/yingjie/Year_3/"
ROOT_FOLDER = os.path.abspath(os.path.join(os.getcwd(), ".."))
import torch
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

sys.path.append(os.path.join(ROOT_FOLDER, "GNN_longterm", "code", "models"))
from GAT import GAT

# %%
# gat = GAT(8, 3, 1, matrix_in=True, batch_size=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

# %%

load_fp = os.path.join(ROOT_FOLDER, "saved", "ACL2018", "dynamic-cooccur", "threshold-0", "weighted_edge-True", "edge_feature-1", "Y_feature-4_Ytype-mean", "node_feature-8_norm-mean_matrix-True_ESwindow-21", "Epoch-100_lr-0.0001_GAT_-20240723-115031.pt")
gat = torch.load(load_fp, map_location='cpu')

# %%
# print(gat.hidden, gat.out_head, gat.dropout)
print(gat)

# %%
a = torch.DoubleTensor([0.9, 0.22])
torch.nn.functional.softmax(a)

# %%
a_ = torch.concatenate([a, a], dim=-1)

# %%
a_.shape, li.shape

# %%
li = torch.DoubleTensor(2, 8, 3)
torch.matmul(a_, li)

# %%
import torch.nn.functional as F

F.softmax(a, dim=2)

# %%
F.softmax(a, dim=2) + a

# %%
torch.DoubleTensor([0.5, 0.5])

# %% [markdown]
# ### Try LLM Generating Relationship
#

# %% [markdown]
#
# #### Qwen2-7B-Instruct

# %%
!pip
install
accelerate

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch

# device = "cuda" # the device to load the model onto
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")


# %%
def generate(prompt, model=model, tokenizer=tokenizer):
    # prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "system", "content": "You are a helpful financial text analyser."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=256
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


# %%
response = generate(
    prompt="""
Given the following text, what could be the strength of relationship between the mentioned firms? Please generate the output in the format of Python dictionary where the key is pair of firms and the value is its strength between -1 to 1, e.g. {("RPT", "TGT") : 0.1}. 

Mentioned Firms: "RPT", "TGT", "FRT", "SPG"

Text: When you get a 1-2 punch of inflation and recessionary fears as we’ve had in 2022, investors begin to lose faith in the ability of shopping centers and regional malls to thrive under adverse conditions. As a result, the prices of retail stocks have been hit hard in recent months. Analysts have been forced to lower their target prices, yet their upside targets from current levels remain high. Here are three retail REITs that analysts have recently cited as having the highest potential upside:RPT Realty (NYSE: RPT) is a NYC-based REIT that owns, operates and manages 57 open air shopping centers across 16 states, mostly in the Northeastern U.S. RPT Realty has a 93.3% overall occupancy rate in its portfolio of stores. This rate has been fairly stable over the last two years.The 52-week range for RPT is $7.51 to $14.99, but the stock has fallen 42% since last November. The stock is currently sitting right near its 52-week low.2nd quarter 2022 revenue declined, but earnings per share (EPS) was up from the previous quarter. RPT Realty pays a $0.52 annual dividend, yielding just above 6%.Raymond James analyst R.J. Milligan recently maintained an Outperform rating on RPT, while lowering the price target to $13 from $15. At a recent price of $7.55, this creates a huge upside potential of 72%.Federal Realty Investment (NYSE: FRT) is another retail REIT, based in North Bethesda, MD. Federal Realty Investment was established in 1962 and is a member of the S&amp;P 500. The REIT owns and operates 105 retail stores in grocery-anchored shopping centers and mixed-use retail centers.Federal Realty prides itself on having the longest record of annual dividend rate increases among U.S. REITs. The 52-week price range is $86.50 to $140.51. Just like RPT Realty, it is currently sitting near its 52-week low and seems to be in a serious downtrend.However, Milligan continues to see value in the stock. He also maintained a Strong Buy on Federal Realty, even while lowering the price target from $140 to $130. Therefore, from its recent price of $88, the analyst’s view would be that Federal Realty now has an upside potential of 47.7%. That’s a lot of ground to make up, but with its long-standing history, an annual dividend of $4.32 and yield of 4.7%, this REIT could be a long-term winner. However, investors may want to see some price stability first.Story continuesSimon Property Group Inc (NYSE: SPG) is one of the largest and most well-known retail REITs. Simon owns and manages shopping centers and premier outlet malls in 37 states and Puerto Rico. It also owns properties in Asia, Canada and Europe. The Indianapolis-based REIT is a member of the S&amp;P 100.Simon Property Group stock dropped significantly in the 2020 COVID-19 market crash, declining from $128 in January to less than $37 by the end of March. Investors who stayed the course were rewarded when SPG rebounded to $161 by November 2021. However, since then, the stock has again languished due to inflation and recessionary concerns. The stock is hitting new lows for the year.Regardless, Richard Hill of Morgan Stanley recently maintained his Overweight position on Simon, while slightly lowering the target price from $133 to $131. This represents about a 51% potential upside from Simon’s recent price of $86.75. Given the REIT’s history of rebounding, and an annual dividend of $7.00 that yields 7.5%, Simon Property Group could be a substantial bargain at this level.Check out: This Little Known REIT Has Produced Double-Digit Annual Returns For The Past Five YearsPlease bear in mind that analysts’ opinions are not always correct, and the best analysts are only right about half the time. Investors should therefore perform their own due diligence when making decisions about what stocks to buy, and simply use the price targets as a guide.Latest Private Market Real Estate Insights:Arrived Homes expanded its offerings to include shares in short-term rental properties with a minimum investment of $100. The platform has already funded over 160 single-family rentals valued at over $60 million.The Flagship Real Estate Fund through Fundrise is up 7.3% year to date and has just added a new rental home community in Charleston, SC to its portfolio.Find more news, insights and offerings on Benzinga Alternative InvestmentsImage by Vintage Tone on ShutterstockSee more from Benzinga3 Health Care REITs With The Highest Upside, According to Analysts3 REITs with the Highest Upside According to AnalystsDon't miss real-time alerts on your stocks - join Benzinga Pro for free! Try the tool that will help you invest smarter, faster, and better.© 2022 Benzinga.com. Benzinga does not provide investment advice. All rights reserved.                   

""")

# %%
print(response)

# %%
## Iterate through daily news in preprocessed_daily_2 folder, generate the response for each row.
## Skip the rows where there is no text.
import os
import pandas as pd

preprocessed_daily_2 = "/home/yingjie_niu/Year_3/GNN_longterm/data/raw/SPNews/news/preprocessed_daily_2/"
for f in os.listdir(preprocessed_daily_2):
    df = pd.read_csv(preprocessed_daily_2 + "2022-09-27.csv", index_col=0)
    df = df[~df['Text'].str.startswith("[""]")]  # skip the rows without "Text"
    df = df[~df['Text'].str.startswith("['']")]  # skip the rows without "Text"
    df.reset_index(drop=True, inplace=True)
    print(df['Text'])
    break

# %%
df['Generated Text'] = df.apply(lambda x: generate(prompt=f"""Given the following text, what could be the strength of relationship between the mentioned firms? Please generate the output in the format of Python dictionary where the key is pair of firms and the value is its strength between -1 to 1, e.g. ("RPT", "TGT") : 0.1. 

Mentioned Firms: {x["Related Company"].strip("[]")}

Text: {x["Text"].strip("[]")} """), axis=1)

# %%
df['Generated Text'] = df['Date']
for i in range(df.shape[0]):
    df['Generated Text'][i] = generate(prompt=f"""Given the following text, what could be the strength of relationship between the mentioned firms? Please generate the output in the format of Python dictionary where the key is pair of firms and the value is its strength between -1 to 1, e.g. ("RPT", "TGT") : 0.1. 

Mentioned Firms: {df["Related Company"][i].strip("[]")}

Text: {df["Text"][i].strip("[]")} """)

# %%
df

# %%
print(df['Text'][2])

# %%
print(df['Generated Text'][2])

# %%
from math import comb

comb(5, 3) * 0.6 ** 3 * 0.4 ** 2 + comb(5, 4) * 0.6 ** 4 * 0.4 ** 1 + comb(5, 5) * 0.6 ** 5

# %%



