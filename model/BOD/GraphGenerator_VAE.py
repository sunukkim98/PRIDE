import torch
import torch.nn as nn
class GraphGenerator_VAE(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.encoder = nn.Linear(emb_size * 2, 64)
        self.relu = nn.ReLU(inplace=True)
        self.fc_encoder = nn.Linear(64, 64)
        self.fc_decode = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h = self.relu(self.encoder(x))
        return self.fc_encoder(h)

    def decode(self, z):
        return self.sigmoid(self.fc_decode(z))

    def forward(self, user_e, item_e):
        x = torch.cat((user_e, item_e), dim=1)
        z = self.encode(x)
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
