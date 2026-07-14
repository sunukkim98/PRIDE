from collections import defaultdict
import csv
import datetime
import json
import os
import random
from typing import Counter
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm


class BasicDataset(Dataset):
    def __init__(self, path, config) -> None:
        super().__init__()
        self.path = path
        self.config = config
        self.min_len = config["min_interaction"]

        self._load_data()

    def __len__(self):
        return self.n_users
    
    def __getitem__(self, index):
        return index
    
    def _load_org_data(self):
        '''
        data format:
        User ID; [Item IDs]
        '''
        # 读取 JSON 文件
        with open(os.path.join(self.path, "data.json"), 'r') as f:
            users = json.load(f)


        valid_users = {user: items for user, items in users.items() if len(items) >= self.config['min_interaction']}

        user_mapping = {user: idx for idx, user in enumerate(valid_users.keys())}
        item_set = {item for items in valid_users.values() for item in items}
        item_mapping = {item: idx for idx, item in enumerate(item_set)}

        user_item_dict = {user_mapping[user]: [item_mapping[item] for item in items] for user, items in valid_users.items()}
        
        
        if not os.path.exists(os.path.join(self.path, "processed")): os.makedirs(f'{self.path}/processed', exist_ok=True)
        with open(os.path.join(self.path, f"processed/user_interactions_more_{self.config['min_interaction']}.pickle"), 'wb') as f:
            pickle.dump((user_item_dict, len(user_mapping), len(item_mapping), item_mapping), f)
        
        return user_item_dict, len(user_mapping), len(item_mapping), item_mapping


    def _load_data(self):
        if os.path.exists(os.path.join(self.path, f"processed/user_interactions_more_{self.config['min_interaction']}.pickle")):
            with open(os.path.join(self.path, f"processed/user_interactions_more_{self.config['min_interaction']}.pickle"), 'rb') as f:
                self.user_interactions, self.n_users, self.n_items, item_map = pickle.load(f)
        else:
            self.user_interactions, self.n_users, self.n_items, item_map = self._load_org_data()
       

    def get_train_batch(self, idx):
        raise NotImplementedError

    def get_val_batch(self, idx):
        raise NotImplementedError

    def get_test_batch(self, idx):
        raise NotImplementedError

    @staticmethod
    def collate_fn(samples):
        return samples


class CFDataset(BasicDataset):
    def __init__(self, path, config) -> None:
        super().__init__(path, config)
        self.split_ratio = [0.7, 0.1, 0.2]
        self._build_set()
    
    def __len__(self):
        if hasattr(self, "n_train_pairs"):
            return self.n_train_pairs
        return self.n_train_num
    
    def _build_set(self):
        self.n_train_num = 0
        self.train_data = [[] for _ in range(self.n_users)]
        self.val_data = [[] for _ in range(self.n_users)]
        self.test_data = [[] for _ in range(self.n_users)]

        all_num = 0

        for user in range(self.n_users):
            random.shuffle(self.user_interactions[user])
            n_inter_items = len(self.user_interactions[user])
            n_train_items = int(n_inter_items * self.split_ratio[0])
            n_test_items = int(n_inter_items * self.split_ratio[2])
            self.train_data[user] += self.user_interactions[user][:n_train_items]
            self.val_data[user] += self.user_interactions[user][n_train_items:-n_test_items]
            self.test_data[user] += self.user_interactions[user][-n_test_items:]
            self.n_train_num += n_train_items
            all_num += n_inter_items
        

        self.avg_inter = int(self.n_train_num / self.n_users)

        self._add_noise(self.config["noise"], self.config["add_p"])

        self.n_train_num = sum(len(items) for items in self.train_data)

        self._build_dcf_train_pairs()

        print(f"#User: {self.n_users}, #Item: {self.n_items}, #Ratings: {all_num}, AvgLen: {int(10 * (all_num / self.n_users)) / 10}, Sparsity: {100 - int(10000 * all_num / (self.n_users * self.n_items)) / 100}")

    def _add_noise(self, noise_ratio, add_p):
        # user_noise = {}
        self.user_noise = {}
        self.noisy_pairs = set()

        if noise_ratio == 0.0:
            return
        for user, interaction in enumerate(self.train_data):
            noisy = int(noise_ratio * (len(interaction)))
            # TODO: 从 range(0, self.n_items) 随机选取 noisy 个不在 interaction 中的 item，拼接到 interaction 后
            if noisy > 0:
                available_items = set(range(self.n_items)) - set(interaction)
                noisy_items = random.sample(available_items, noisy)
                
                self.user_noise[user] = noisy_items
                for item in noisy_items:
                    self.noisy_pairs.add((user, item))

                # 拼接到 interaction 后
                interaction.extend(noisy_items)
                self.train_data[user] = interaction

    def _build_dcf_train_pairs(self):
        """
        DCF-BPR용 train pair pool 구성
        sample_id == self.train_pairs index
        """
        self.train_pairs = []
        self.active_pair_mask = []

        for user in range(self.n_users):
            for item in self.train_data[user]:
                is_noisy = 0
                if hasattr(self, "noisy_pairs") and (user, item) in self.noisy_pairs:
                    is_noisy = 1
                
                self.train_pairs.append({
                    "user": user,
                    "item": item,
                    "is_noisy": is_noisy
                })
                self.active_pair_mask.append(True)
        
        self.active_pair_mask = np.array(self.active_pair_mask, dtype=bool)
        self.n_train_pairs = len(self.train_pairs)

    def get_active_train_size(self):
        if hasattr(self, "active_pair_mask"):
            return int(self.active_pair_mask.sum())
        return self.n_train_num

    def get_interaction_matrix(self, device):
        user_list = []
        item_list = []
        for user in range(self.n_users):
            items = self.train_data[user]
            for item in items:
                user_list.append(user)
                item_list.append(item)
        user_dim = torch.tensor(user_list, device=device)
        item_dim = torch.tensor(item_list, device=device) 
        index = torch.stack((user_dim, item_dim))
        data = torch.ones(index.size(-1), device=device)
        Graph = torch.sparse_coo_tensor(index, data, torch.Size([self.n_users, self.n_items]), device=device)
        return Graph 
    
    def get_interaction_matrix_dcf(self, device):
        user_list = []
        item_list = []

        for sid, pair in enumerate(self.train_pairs):
            if hasattr(self, "active_pair_mask") and not self.active_pair_mask[sid]:
                continue
            user_list.append(pair["user"])
            item_list.append(pair["item"])

        user_dim = torch.tensor(user_list, device=device)
        item_dim = torch.tensor(item_list, device=device)
        index = torch.stack((user_dim, item_dim))
        data = torch.ones(index.size(-1), device=device)

        graph = torch.sparse_coo_tensor(
            index, data, torch.Size([self.n_users, self.n_items]), device=device
        )
        return graph

    def get_train_batch(self, inter_list, multi_sample=False, k=5):
        inter_list = inter_list.squeeze().tolist()
        pos_item_list = []
        neg_item_list = []
        is_noisy_list = []
        user_list = np.random.randint(0, self.n_users, len(inter_list))
        if multi_sample:
            for user in user_list:
                if len(self.train_data[user]) >= k:
                    pos_items = np.random.choice(self.train_data[user], k, replace=False)
                else:
                    pos_items = np.random.choice(self.train_data[user], k, replace=True)
                pos_item_list.append(pos_items)

                # multi_sample에서는 pos_items가 여러 개이므로 각 item의 noisy 여부를 기록
                if hasattr(self, "noisy_pairs"):
                    noisy_flags = [int((user, item) in self.noisy_pairs) for item in pos_items]
                else:
                    noisy_flags = [0] * len(pos_items)
                is_noisy_list.append(noisy_flags)

                neg_item = random.randint(0, self.n_items-1)
                while neg_item in self.train_data[user]:
                    neg_item = random.randint(0, self.n_items-1)
                neg_item_list.append(neg_item)
        else:
            for user in user_list:
                pos_item = np.random.choice(self.train_data[user])
                pos_item_list.append(pos_item)

                # sampled positive가 synthetic noise인지 확인
                if hasattr(self, "noisy_pairs"):
                    is_noisy_list.append(int((user, pos_item) in self.noisy_pairs))
                else:
                    is_noisy_list.append(0)
                
                neg_item = random.randint(0, self.n_items-1)
                while neg_item in self.train_data[user]:
                    neg_item = random.randint(0, self.n_items-1)
                neg_item_list.append(neg_item)
        return user_list, np.array(pos_item_list), np.array(neg_item_list), np.array(is_noisy_list)
    
    def get_train_batch_dcf(self, batch_idx_list):
        """
        DCF-BPR 전용 batch 생성 함수
        return:
            sample_ids, user_list, pos_item_list, neg_item_list, is_noisy_list
        """
        if isinstance(batch_idx_list, torch.Tensor):
            batch_idx_list = batch_idx_list.squeeze().tolist()
        
        if isinstance(batch_idx_list, int):
            batch_idx_list = [batch_idx_list]

        sample_ids = []
        user_list = []
        pos_item_list = []
        neg_item_list = []
        is_noisy_list = []

        if hasattr(self, "active_pair_mask"):
            active_ids = np.where(self.active_pair_mask)[0]
            if len(active_ids) == 0:
                raise ValueError("No active training pairs available.")
        
        for sid in batch_idx_list:
            sid = int(sid)

            if hasattr(self, "active_pair_mask") and not self.active_pair_mask[sid]:
                sid = int(np.random.choice(active_ids))

            pair = self.train_pairs[sid]
            user = pair["user"]
            pos_item = pair["item"]
            is_noisy = pair["is_noisy"]

            neg_item = random.randint(0, self.n_items-1)
            while neg_item in self.train_data[user]:
                neg_item = random.randint(0, self.n_items-1)

            sample_ids.append(sid)
            user_list.append(user)
            pos_item_list.append(pos_item)
            neg_item_list.append(neg_item)
            is_noisy_list.append(is_noisy)

        return (
            torch.tensor(sample_ids, dtype=torch.long),
            torch.tensor(user_list, dtype=torch.long),
            torch.tensor(pos_item_list, dtype=torch.long),
            torch.tensor(neg_item_list, dtype=torch.long),
            torch.tensor(is_noisy_list, dtype=torch.long)
        )
    
    def mark_as_relabelled(self, relabel_ids):
        """
        DCF-BPR setting:
        noisy positive interaction을 다음 에폭부터 positive pool에서 제외
        """
        if not hasattr(self, "active_pair_mask"):
            return
        
        for sid in relabel_ids:
            sid = int(sid)
            if 0 <= sid < len(self.active_pair_mask):
                self.active_pair_mask[sid] = False

    def get_val_batch(self, user_list):
        return np.array(user_list), [self.val_data[user] for user in user_list], [self.train_data[user] for user in user_list]
    
    def get_test_batch(self, user_list):
        return np.array(user_list), [self.test_data[user] for user in user_list], [self.train_data[user] + self.val_data[user] for user in user_list]

    def gcn_graph(self):
        user_list = []
        item_list = []
        for user in range(self.n_users):
            items = self.train_data[user]
            for item in items:
                user_list.append(user)
                item_list.append(item)

        user_dim = torch.LongTensor(user_list)
        item_dim = torch.LongTensor(item_list)

        first_sub = torch.stack([user_dim, item_dim + self.n_users])
        second_sub = torch.stack([item_dim + self.n_users, user_dim])
        index = torch.cat([first_sub, second_sub], dim=1)  # [2, 2*E]
        value = torch.ones(index.size(1))

        N = self.n_users + self.n_items
        graph = torch.sparse_coo_tensor(index, value, torch.Size([N, N]))

        # Degree 계산 (sparse 방식)
        deg = torch.sparse.sum(graph, dim=1).to_dense()  # [N]
        deg[deg == 0] = 1  # divide-by-zero 방지
        deg_inv_sqrt = torch.pow(deg, -0.5)

        # edge-wise normalization
        row, col = index
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        norm_graph = torch.sparse_coo_tensor(index, norm, torch.Size([N, N])).coalesce()
        return norm_graph
    
    def gcn_graph_dcf(self):
        user_list = []
        item_list = []

        for sid, pair in enumerate(self.train_pairs):
            if hasattr(self, "active_pair_mask") and not self.active_pair_mask[sid]:
                continue
            user_list.append(pair["user"])
            item_list.append(pair["item"])

        user_dim = torch.LongTensor(user_list)
        item_dim = torch.LongTensor(item_list)

        first_sub = torch.stack([user_dim, item_dim + self.n_users])
        second_sub = torch.stack([item_dim + self.n_users, user_dim])
        index = torch.cat([first_sub, second_sub], dim=1)
        value = torch.ones(index.size(1))

        N = self.n_users + self.n_items
        graph = torch.sparse_coo_tensor(index, value, torch.Size([N, N]))

        deg = torch.sparse.sum(graph, dim=1).to_dense()
        deg[deg == 0] = 1
        deg_inv_sqrt = torch.pow(deg, -0.5)

        row, col = index
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        norm_graph = torch.sparse_coo_tensor(index, norm, torch.Size([N, N])).coalesce()
        return norm_graph