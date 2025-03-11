import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as gnn
from torch_geometric.nn import GraphConv
from torch_geometric.data import HeteroData
from torch_geometric.loader import HeteroGraphSAINTRandomWalkSampler, HeteroGraphSAINTNodeSampler

device = 'cuda' if torch.cuda.is_available() else 'cpu' 

# Initialize the HeteroData object
data = HeteroData()

USER_NODES = 1000
TWEET_NODES = 2000
LIST_NODES = 1000

save_path = osp.join(osp.dirname(osp.realpath(__file__)), 'processed', 'IMDB')

# Define the 'user' node type
data['user'].num_nodes = USER_NODES
data['user'].num = torch.randn(USER_NODES, 11).float()  
data['user'].cat = torch.randint(0, 10, (USER_NODES, 7)).float()  
data['user'].desc = torch.randn(USER_NODES, 768)  
data['user'].y = torch.randint(0, 2, (USER_NODES, 2))  

# Define the 'tweet' node type
data['tweet'].num = torch.randint(0, 10, (TWEET_NODES, 14)).float()
data['tweet'].text = torch.randn(TWEET_NODES, 768) 
data['tweet'].num_nodes = TWEET_NODES
data['tweet'].cat = torch.randint(0, 10, (TWEET_NODES, 128)).float()

# Define the 'list' node type
data['list'].num = torch.randint(0, 10, (LIST_NODES, 5)).float()
data['list'].desc = torch.randn(LIST_NODES, 768) 
data['list'].num_nodes = LIST_NODES  # Additional list features


def get_edges(num_nodes, num_edges=1000):
    return torch.randint(0, num_nodes, (2, num_edges))



# Define edges
data[('user', 'following', 'user')].edge_index = get_edges(USER_NODES)
data[('tweet', 'post', 'user')].edge_index = get_edges(USER_NODES)
data[('list', 'following', 'user')].edge_index = get_edges(LIST_NODES)
data[('list', 'own', 'user')].edge_index = get_edges(LIST_NODES)
data[('list', 'member', 'user')].edge_index = get_edges(LIST_NODES)
data[('tweet', 'like', 'user')].edge_index = get_edges(USER_NODES)
data[('tweet', 'pin', 'user')].edge_index = get_edges(USER_NODES)
data[('user', 'rev_following', 'user')].edge_index = get_edges(USER_NODES)
data[('user', 'rev_post', 'tweet')].edge_index = get_edges(USER_NODES)
data[('user', 'rev_like', 'tweet')].edge_index = get_edges(USER_NODES)
data[('user', 'rev_pin', 'tweet')].edge_index = get_edges(USER_NODES)
data[('user', 'rev_following', 'list')].edge_index = get_edges(LIST_NODES)
data[('user', 'rev_member', 'list')].edge_index = get_edges(LIST_NODES)
data[('user', 'rev_own', 'list')].edge_index = get_edges(LIST_NODES)

# Print data summary
print(data)


train_loader = HeteroGraphSAINTNodeSampler(
    data, batch_size={'user': 100, 'tweet': 200, 'list': 100},
    num_steps=1, sample_coverage=100,
    num_workers=0
)

sample_batch = next(iter(train_loader))


class BotRGCN(nn.Module):
    def __init__(self, 
                 user_desc_size = 768, user_num_size = 11, user_cat_size = 7,
                 tweet_num_size = 14, tweet_text_size = 768,
                 list_num_size = 5, list_desc_size = 768,
                 h = 120, dropout1 = 0.5, dropout2 = 0.2):
        
        super(BotRGCN, self).__init__()
        
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        
        self.leaky_relu = nn.LeakyReLU()
        self.user_conv1 = GraphConv(h, h)
        self.user_conv2 = GraphConv(h, h)
        self.user_conv3 = GraphConv(h, h)
        self.user_conv4 = GraphConv(h, h)
        self.user_conv5 = GraphConv(h, h)
        self.user_conv6 = GraphConv(h, h)
        self.user_conv7 = GraphConv(h, h)
        self.user_conv8 = GraphConv(h, h)
        
        self.tweet_conv1 = GraphConv(h, h * 3 // 4)
        self.tweet_conv2 = GraphConv(h, h * 3 // 4)
        self.tweet_conv3 = GraphConv(h, h * 3 // 4)
        
        self.list_conv1 = GraphConv(h, h // 2)
        self.list_conv2 = GraphConv(h, h // 2)
        self.list_conv3 = GraphConv(h, h // 2)
        self.linear_relu_user_desc = nn.Sequential(
            nn.Linear(user_desc_size, h // 3),
            nn.LeakyReLU()
        )
        self.linear_relu_user_num = nn.Sequential(
            nn.Linear(user_num_size, h // 3),
            nn.LeakyReLU()
        )
        self.linear_relu_user_cat = nn.Sequential(
            nn.Linear(user_cat_size, h // 3),
            nn.LeakyReLU()
        )
        self.linear_relu_user_input = nn.Sequential(
            nn.Linear(h, h),
            nn.LeakyReLU()
        )
        
        self.linear_relu_tweet_num = nn.Sequential(
            nn.Linear(tweet_num_size, h // 4),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet_text = nn.Sequential(
            nn.Linear(tweet_text_size, h // 2),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet_input = nn.Sequential(
            nn.Linear(h * 3 // 4, h * 3 // 4),
            nn.LeakyReLU()
        )
        
        self.linear_relu_list_num = nn.Sequential(
            nn.Linear(list_num_size, h // 4),
            nn.LeakyReLU()
        )
        self.linear_relu_list_desc = nn.Sequential(
            nn.Linear(list_desc_size, h // 4),
            nn.LeakyReLU()
        )
        self.linear_relu_list_input = nn.Sequential(
            nn.Linear(h // 2, h // 2),
            nn.LeakyReLU()
        )
        
        self.user_sage11 = gnn.SAGEConv((h, h), h, root_weight = False)
        self.user_sage12 = gnn.SAGEConv((h, h), h, root_weight = False)
        self.user_sage13 = gnn.SAGEConv((h * 3 // 4, h), h, root_weight = False)
        self.user_sage14 = gnn.SAGEConv((h * 3 // 4, h), h, root_weight = False)
        self.user_sage15 = gnn.SAGEConv((h * 3 // 4, h), h, root_weight = False)
        self.user_sage16 = gnn.SAGEConv((h // 2, h), h, root_weight = False)
        self.user_sage17 = gnn.SAGEConv((h // 2, h), h, root_weight = False)
        self.user_sage18 = gnn.SAGEConv((h // 2, h), h, root_weight = False)
        
        self.user_self1 = nn.Linear(h, h // 2)
        self.user_neigh1 = nn.Linear(h, h // 2)
        
        self.tweet_sage11 = gnn.SAGEConv((h, h * 3 // 4), h * 3 // 4, root_weight = False)
        self.tweet_sage12 = gnn.SAGEConv((h, h * 3 // 4), h * 3 // 4, root_weight = False)
        self.tweet_sage13 = gnn.SAGEConv((h, h * 3 // 4), h * 3 // 4, root_weight = False)
        
        self.tweet_self1 = nn.Linear(h * 3 // 4, h * 3 // 8)
        self.tweet_neigh1 = nn.Linear(h * 3 // 4, h * 3 // 8)
        
        self.list_sage11 = gnn.SAGEConv((h, h // 2), h // 2, root_weight = False)
        self.list_sage12 = gnn.SAGEConv((h, h // 2), h // 2, root_weight = False)
        self.list_sage13 = gnn.SAGEConv((h, h // 2), h // 2, root_weight = False)
        
        self.list_self1 = nn.Linear(h // 2, h // 4)
        self.list_neigh1 = nn.Linear(h // 2, h // 4)

        self.linear_relu_user_output = nn.Sequential(
            nn.Linear(h, h),
            nn.LeakyReLU()
        )
        
        self.linear_prediction = nn.Linear(h, 2)
        
    def forward(self, graph):
        
        user_desc = self.linear_relu_user_desc(graph['user'].desc)
        user_num = self.linear_relu_user_num(graph['user'].num)
        user_cat = self.linear_relu_user_cat(graph['user'].cat)
        
        user_x = torch.cat((user_desc, user_num, user_cat), dim = 1)
        user_x = self.linear_relu_user_input(user_x)
        user_x = F.normalize(user_x)

        tweet_num = self.linear_relu_tweet_num(graph['tweet'].num)
        tweet_text = self.linear_relu_tweet_text(graph['tweet'].text)
        
        tweet_x = torch.cat((tweet_num, tweet_text), dim = 1)
        tweet_x = self.linear_relu_tweet_input(tweet_x)
        tweet_x = F.normalize(tweet_x)
        
        list_num = self.linear_relu_list_num(graph['list'].num)
        list_desc = self.linear_relu_list_desc(graph['list'].desc)
        
        list_x = torch.cat((list_num, list_desc), dim = 1)
        list_x = self.linear_relu_list_input(list_x)
        list_x = F.normalize(list_x)
        
        user_x1 = self.user_sage11((user_x, user_x), graph['user', 'following', 'user'].edge_index)
        user_x2 = self.user_sage12((user_x, user_x), graph['user', 'rev_following', 'user'].edge_index)
        user_x3 = self.user_sage13((tweet_x, user_x), graph['tweet', 'post', 'user'].edge_index)
        user_x4 = self.user_sage14((tweet_x, user_x), graph['tweet', 'like', 'user'].edge_index)
        user_x5 = self.user_sage15((tweet_x, user_x), graph['tweet', 'pin', 'user'].edge_index)
        user_x6 = self.user_sage16((list_x, user_x), graph['list', 'member', 'user'].edge_index)
        user_x7 = self.user_sage17((list_x, user_x), graph['list', 'following', 'user'].edge_index)
        user_x8 = self.user_sage18((list_x, user_x), graph['list', 'own', 'user'].edge_index)
        
        tweet_x1 = self.tweet_sage11((user_x, tweet_x), graph['user', 'rev_post', 'tweet'].edge_index)
        tweet_x2 = self.tweet_sage12((user_x, tweet_x), graph['user', 'rev_like', 'tweet'].edge_index)
        tweet_x3 = self.tweet_sage13((user_x, tweet_x), graph['user', 'rev_pin', 'tweet'].edge_index)
        
        list_x1 = self.list_sage11((user_x, list_x), graph['user', 'rev_following', 'list'].edge_index)
        list_x2 = self.list_sage12((user_x, list_x), graph['user', 'rev_member', 'list'].edge_index)
        list_x3 = self.list_sage13((user_x, list_x), graph['user', 'rev_own', 'list'].edge_index)
        
        user_self = self.user_self1(user_x)
        user_neigh = user_x1 + user_x2 + user_x3 + user_x4 + user_x5 + user_x6 + user_x7 + user_x8
        user_neigh = self.user_neigh1(user_neigh)
        user_x = torch.cat((user_self, user_neigh), dim = 1)
        user_x = self.leaky_relu(user_x)
        user_x = F.normalize(user_x)
        user_x = F.dropout(user_x, p = self.dropout1, training = self.training)
        
        tweet_self = self.tweet_self1(tweet_x)
        tweet_neigh = tweet_x1 + tweet_x2 + tweet_x3
        tweet_neigh = self.tweet_neigh1(tweet_neigh)
        tweet_x = torch.cat((tweet_self, tweet_neigh), dim = 1)
        tweet_x = self.leaky_relu(tweet_x)
        tweet_x = F.normalize(tweet_x)
        tweet_x = F.dropout(tweet_x, p = self.dropout1, training = self.training)
        
        list_self = self.list_self1(list_x)
        list_neigh = list_x1 + list_x2 + list_x3
        list_neigh = self.list_neigh1(list_neigh)
        list_x = torch.cat((list_self, list_neigh), dim = 1)
        list_x = self.leaky_relu(list_x)
        list_x = F.normalize(list_x)
        list_x = F.dropout(list_x, p = self.dropout1, training = self.training)
        
        user_x1 = self.user_sage11((user_x, user_x), graph['user', 'following', 'user'].edge_index)
        user_x2 = self.user_sage12((user_x, user_x), graph['user', 'rev_following', 'user'].edge_index)
        user_x3 = self.user_sage13((tweet_x, user_x), graph['tweet', 'post', 'user'].edge_index)
        user_x4 = self.user_sage14((tweet_x, user_x), graph['tweet', 'like', 'user'].edge_index)
        user_x5 = self.user_sage15((tweet_x, user_x), graph['tweet', 'pin', 'user'].edge_index)
        user_x6 = self.user_sage16((list_x, user_x), graph['list', 'member', 'user'].edge_index)
        user_x7 = self.user_sage17((list_x, user_x), graph['list', 'following', 'user'].edge_index)
        user_x8 = self.user_sage18((list_x, user_x), graph['list', 'own', 'user'].edge_index)
        
        user_self = self.user_self1(user_x)
        user_neigh = user_x1 + user_x2 + user_x3 + user_x4 + user_x5 + user_x6 + user_x7 + user_x8
        user_neigh = self.user_neigh1(user_neigh)
        user_x = torch.cat((user_self, user_neigh), dim = 1)
        user_x = self.leaky_relu(user_x)
        user_x = F.normalize(user_x)
        user_x = F.dropout(user_x, p = self.dropout2, training = self.training)
        
        user_x = self.linear_relu_user_output(user_x)
        user_x = self.linear_prediction(user_x)
            
        return user_x
    

sample_batch.to(device)
print(sample_batch)
model = BotRGCN().to(device)
# print(model)
out = model(sample_batch)
# print(out)