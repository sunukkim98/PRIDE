import torch
import torch.nn as nn
class GraphGenerator_VAE(nn.Module):
    def __init__(self, data, emb_size):
        super(GraphGenerator_VAE, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.encoder = nn.Linear(self.latent_size*2, 32)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fc_encoder = nn.Linear(32, 32)
        self.fc_encoder_mu = nn.Linear(32, 16)
        self.fc_encoder_var = nn.Linear(32, 16)
        self.fc_reparameterize = nn.Linear(16, 32)
        self.fc_decode = nn.Linear(32, 1)

    def encode(self, x):
        output = self.encoder(x)
        h = self.relu(output)
        return self.fc_encoder(h)
 
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.rand_like(std)
        return mu + eps * std
 
    def decode(self, z):
        return self.sigmoid(self.fc_decode(z))

    def forward(self, user_e, item_e):
        input_vec = torch.cat((user_e,item_e), axis = 1)
        z = self.encode(input_vec)
        return self.decode(z)

import torch
import torch.nn as nn

class GraphGenerator_2MLP(nn.Module):
    """
    2-MLP 구조의 그래프 파라미터 생성기
    입력: 사용자 및 아이템 임베딩 벡터 (batch_size x emb_size)
    출력: weight scalar (batch_size x 1)
    """
    def __init__(self, emb_size):
        super(GraphGenerator_2MLP, self).__init__()
        hidden_size = emb_size // 2 
        self.net = nn.Sequential(
            nn.Linear(emb_size * 2, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, user_e: torch.Tensor, item_e: torch.Tensor) -> torch.Tensor:
        # user_e, item_e: (batch_size, emb_size)
        x = torch.cat([user_e, item_e], dim=1)  # (batch_size, emb_size*2)
        weight = self.net(x)  # (batch_size, 1), in (0,1)
        return weight.squeeze(dim=1)  # (batch_size,)
