import json
import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import DataLoader
from datetime import datetime
import torch.nn.functional as F
from utls.mydataset import CFDataset
from model.MF.MF import *
from model.LightGCN.LightGCN import *
from model.NeuMF.NeuMF import *
from utls.utilize import slice_lists, batch_split
from vector_quantize_pytorch.residual_vq import ResidualVQ
import copy
from model.BOD.GraphGenerator_VAE import GraphGenerator_VAE, GraphGenerator_2MLP
from monitor import Monitor
import matplotlib.pyplot as plt
from collections import defaultdict, deque

class BasicTrainer:
    def __init__(self, trainer_config) -> None:
        self.config = trainer_config
        self.device = trainer_config['device']
        self.n_epochs = trainer_config['n_epochs']
        self.min_epochs = trainer_config['min_epochs']
        self.max_patience = trainer_config.get('patience', 50)
        self.val_interval = trainer_config.get('val_interval', 1)
        self.monitor = Monitor()
        self._codebook_reinit_history = []

    def _log_codebook_reinit(self, tag=None):
        """Log how many codebook entries were reinitialized (dead-code expiry) on the
        last `self.codebook(...)` forward pass, plus the running average across all
        such updates so far."""
        per_layer = self.codebook.num_expired_codes_per_layer
        total = sum(per_layer)
        self._codebook_reinit_history.append(total)
        running_avg = sum(self._codebook_reinit_history) / len(self._codebook_reinit_history)

        label = f" [{tag}]" if tag else ""
        print(
            f"Codebook reinit{label}: {total} entries reinitialized (per-layer: {per_layer}), "
            f"running avg over {len(self._codebook_reinit_history)} updates: {running_avg:.2f}"
        )
        self.monitor.log({
            "codebook_reinit_total": total,
            "codebook_reinit_running_avg": running_avg,
            **{f"codebook_reinit_layer{i}": c for i, c in enumerate(per_layer)},
        })

    def _create_dataset(self, path):
        raise NotImplementedError
    
    def _create_dataloader(self):
        self.dataloader = DataLoader(self.dataset, batch_size=int(self.config["batch_size"]), shuffle=True)
        print(f"Create Dataloader with batch_size:{int(self.config['batch_size'])}")

    def _create_model(self):
        raise NotImplementedError
    
    def _create_opt(self):
        raise NotImplementedError

    def _train_epoch(self, epoch):
        raise NotImplementedError
    
    def _eval_model(self, epoch):
        raise NotImplementedError

    def _save_model(self, best_model_path):
        torch.save({
            'model': self.model.state_dict(),
        }, best_model_path)
    
    def _load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model'])

    def train(self, path=None):
        patience = self.config["patience"]
        best_metrics = -1
        
        best_model_path = f"{self.config['checkpoints']}/{self.config['model']}/{self.config['method']}/{self.config['dataset']}"
        if self.config["main_file"] != "":
            best_model_path = os.path.join(best_model_path, self.config["main_file"])
        if path is not None:
            best_model_path = path
        if not os.path.exists(best_model_path):
            os.makedirs(best_model_path, exist_ok=True)
        best_model_path = os.path.join(best_model_path, f"{self.config['noise']}_{self.config['seed']}.pth")

        self.update_flag = False
        for epoch in range(self.n_epochs):
            self._train_epoch(epoch)
            if (epoch + 1) % self.config["val_interval"] == 0:
                metrics_list, ndcg_list = self._eval_model(epoch)
                metrics = metrics_list[0]
                if (epoch + 1) >= self.config["min_epochs"]:
                    if metrics > best_metrics:
                        best_metrics = metrics
                        # Save the best model
                        self._save_model(best_model_path)
                        patience = self.config["patience"]
                    else:
                        patience -= 1
                        if patience <= 0:
                            print('Early stopping!')
                            break

        self._load_model(best_model_path)
        # Test
        avg_hr, avg_ndcg = self._eval_model(eval_type='test')

        return avg_hr, avg_ndcg

class CFTrainer(BasicTrainer):
    def __init__(self, trainer_config) -> None:
        super().__init__(trainer_config)
    
        self._create_dataset(f"data/{trainer_config['dataset']}")
        self._create_dataloader()
        self._create_model()
        self._create_opt()

    def _create_dataset(self, path):
        self.dataset = CFDataset(path, self.config)
    
    def _create_dataloader(self):
        return super()._create_dataloader()

    def _create_opt(self):
        self.opt = optim.AdamW(self.model.parameters(), lr=self.config["lr"])

    def _create_model(self):
        glo = globals()
        self.config["model_config"]["n_users"] = self.dataset.n_users
        self.config["model_config"]["n_items"] = self.dataset.n_items
        self.model = glo[f'{self.config["model"]}'](self.config["model_config"], self.dataset)
        if torch.cuda.is_available() and self.config["use_gpu"]:
            self.model.cuda()
    
    def _train_epoch(self, epoch):
        start_t = time.time()
        epoch_loss = 0

        for batch_data in self.dataloader:
            self.opt.zero_grad()
            self.model.train()
            user_id_list, pos_item_list, neg_item_list, _ = self.dataset.get_train_batch(batch_data)
            if self.config["model"] in ["NeuMF"]:
                pos_logits, neg_logits, _, _, _, l2_norm_sq = self.model(user_id_list, pos_item_list, neg_item_list)
                pos_logits = torch.sum(pos_logits, dim=1)
                neg_logits = torch.sum(neg_logits, dim=1)
            else:
                users_emb, pos_items_emb, neg_items_emb, l2_norm_sq = self.model(user_id_list, pos_item_list, neg_item_list)
                pos_logits = torch.sum(users_emb * pos_items_emb, dim=1)
                neg_logits = torch.sum(users_emb * neg_items_emb, dim=1)
            loss = self._rec_loss(pos_logits, neg_logits).mean() + self.config["weight_decay"] * l2_norm_sq
            loss.backward()
            self.opt.step()
            epoch_loss += loss.item()
        
        end_t = time.time()
        print(f"Epoch {epoch}: Rec Loss: {epoch_loss/len(self.dataloader):.4f}, Time: {end_t-start_t:.2f}")

    def _rec_loss(self, pos, neg):
        return F.softplus(neg - pos)

    def _eval_model(self, epoch=0, eval_type='val'):
        start_t = time.time()

        assert eval_type in ['val', 'test']
        self.model.eval()
        top_ks = self.config["rec_top_k"]

        recall_list = [0.0 for _ in self.config["rec_top_k"]]
        ndcg_list = [0.0 for _ in self.config["rec_top_k"]]

        user_list = list(range(self.dataset.n_users))
        for batch_data in batch_split(users=user_list, batch_size=self.config["test_batch_size"]):
            if eval_type == 'val':
                user_id_list, user_inter_list, user_train_list = self.dataset.get_val_batch(batch_data)
            else:
                user_id_list, user_inter_list, user_train_list = self.dataset.get_test_batch(batch_data)

            with torch.no_grad():
                score_list = self.model.predict(user_id_list).to(self.device)  # (B, num_items)

            for idx, user_train_items in enumerate(user_train_list):
                if len(user_train_items) > 0:
                    train_items_tensor = torch.tensor(user_train_items, dtype=torch.long, device=self.device)
                    score_list[idx].index_fill_(0, train_items_tensor, float('-inf'))  # GPU-safe masking

            max_k = max(top_ks)

            for user_idx, user_inter_items in enumerate(user_inter_list):
                gt_set = set(user_inter_items)
                _, top_indices = torch.topk(score_list[user_idx], max_k)
                top_indices = top_indices.tolist()

                for idx, k in enumerate(top_ks):
                    top_k = top_indices[:k]

                    num_hits = sum([1 for item in top_k if item in gt_set])
                    recall_k = num_hits / len(gt_set) if gt_set else 0.0

                    dcg = sum([1 / np.log2(i + 2) for i, item in enumerate(top_k) if item in gt_set])
                    idcg = sum([1.0 / np.log2(i + 2) for i in range(len(gt_set))])
                    ndcg_k = dcg / idcg if idcg > 0 else 0.0

                    recall_list[idx] += recall_k
                    ndcg_list[idx] += ndcg_k

        avg_hr = [hr / self.dataset.n_users for hr in recall_list]
        avg_ndcg = [ndcg / self.dataset.n_users for ndcg in ndcg_list]

        end_t = time.time()
        print(("Validation - " if eval_type == 'val' else "Test - ") + f"Time: {end_t - start_t:.2f}")

        epoch_text = f"at Epoch {epoch}" if eval_type == 'val' else ""
        self._print_performance("Recommendation Performance" + epoch_text, ("Recall", "NDCG"), avg_hr, avg_ndcg, self.config["rec_top_k"], eval_type=eval_type)

        return recall_list, ndcg_list

    def _print_performance(self, title, metrics, m1_list, m2_list, top_k_list, eval_type):
        out_text = f"{title}:"
        for i, k in enumerate(top_k_list):
            out_text += f"\n{metrics[0]}@{k}: {m1_list[i]:.4f}, {metrics[1]}@{k}: {m2_list[i]:.4f};"
            if eval_type == 'val':
                self.monitor.log({
                    f"valid_{metrics[0]}@{k}": m1_list[i],
                    f"valid_{metrics[1]}@{k}": m2_list[i]
                })
            else:
                self.monitor.log({
                    f"test_{metrics[0]}@{k}": m1_list[i],
                    f"test_{metrics[1]}@{k}": m2_list[i]
                })
        print(out_text)

class PLDCFTrainer(CFTrainer):
    def __init__(self, trainer_config) -> None:
        super().__init__(trainer_config)
        
        self.begin_adv = trainer_config["model_config"]['denoise_config']['begin_adv']
        self.temp = self.config["model_config"]['denoise_config']["temperature"]
        
    def _train_epoch(self, epoch):
        start_t = time.time()
        epoch_loss = 0
        

        for batch_data in self.dataloader:
            self.opt.zero_grad()
            self.model.train()
            
            if epoch >= self.begin_adv:
                user_id_list, pos_item_list, neg_item_list, is_noisy_list = self.dataset.get_train_batch(batch_data, multi_sample=True, k=self.config["model_config"]['denoise_config']["item_num"])
                
                # pre-selection
                with torch.no_grad():
                    pos_item_list_flat = torch.tensor(pos_item_list).flatten()
                    user_id_list_repeated = torch.tensor(user_id_list).repeat_interleave(self.config["model_config"]['denoise_config']["item_num"])
                    neg_item_list_repeated = torch.tensor(neg_item_list).repeat_interleave(self.config["model_config"]['denoise_config']["item_num"])
                    
                    if self.config["model"] in ["NeuMF"]:
                        pos_logits, neg_logits, _, _, _, _ = self.model(user_id_list_repeated, pos_item_list_flat, neg_item_list_repeated)
                        pos_logits = torch.sum(pos_logits, dim=1)
                        neg_logits = torch.sum(neg_logits, dim=1)
                    else:
                        users_emb, pos_items_emb, neg_items_emb, _ = self.model(user_id_list_repeated, pos_item_list_flat, neg_item_list_repeated)
                        pos_logits = torch.sum(users_emb * pos_items_emb, dim=1)
                        neg_logits = torch.sum(users_emb * neg_items_emb, dim=1)
                    
         
                    temp_loss = self._rec_loss(pos_logits, neg_logits).detach()

                    temp_loss = temp_loss.view(len(user_id_list), self.config["model_config"]['denoise_config']["item_num"])

                    sampling_probabilities = torch.nn.functional.softmax(-temp_loss / self.temp, dim=1)

                    sampled_indices = torch.multinomial(sampling_probabilities, 1, replacement=False).squeeze()

                    pos_item_list = torch.gather(torch.tensor(pos_item_list).to(sampled_indices.device), 1, sampled_indices.unsqueeze(-1)).squeeze().cpu().tolist()
                    
            else:
                user_id_list, pos_item_list, neg_item_list, is_noisy_list = self.dataset.get_train_batch(batch_data)
            
            if self.config["model"] in ["NeuMF"]:
                pos_logits, neg_logits, _, _, _, l2_norm_sq = self.model(user_id_list, pos_item_list, neg_item_list)
                pos_logits = torch.sum(pos_logits, dim=1)
                neg_logits = torch.sum(neg_logits, dim=1)
            else:
                users_emb, pos_items_emb, neg_items_emb, l2_norm_sq = self.model(user_id_list, pos_item_list, neg_item_list)
                pos_logits = torch.sum(users_emb * pos_items_emb, dim=1)
                neg_logits = torch.sum(users_emb * neg_items_emb, dim=1)
            loss = self._rec_loss(pos_logits, neg_logits).mean() + self.config["weight_decay"] * l2_norm_sq
            loss.backward()
            self.opt.step()
            epoch_loss += loss.item()
        
        end_t = time.time()
        print(f"Epoch {epoch}: Rec Loss: {epoch_loss/len(self.dataloader):.4f}, Time: {end_t-start_t:.2f}")


class BODCFTrainer(CFTrainer):
    def __init__(self, trainer_config):
        super().__init__(trainer_config)

        # BOD-BPR config
        self.generator_lr = 0.001
        self.generator_reg = 0.0001
        self.weight_bpr = 1
        self.weight_alignment = 1.0
        self.weight_uniformity = 1.0
        self.outer_loop = 1
        self.inner_loop = 1
        self.alpha = self.config["alpha"]
        self.gamma = self.config["gamma"]

        self.model_generator = GraphGenerator_2MLP(emb_size=64).to(self.device)
        self.generator_opt = torch.optim.Adam(self.model_generator.parameters(), lr=self.generator_lr)
        self.model_parameters = list(self.model.parameters())

    def bpr_loss_weight(self, user_emb, pos_item_emb, neg_item_emb, weight_pos, weight_neg):
        pos_score = weight_pos * torch.mul(user_emb, pos_item_emb).sum(dim=1)
        neg_score = weight_neg * torch.mul(user_emb, neg_item_emb).sum(dim=1)
        loss = -torch.log(10e-8 + torch.sigmoid(pos_score - neg_score))
        return torch.mean(loss)

    def alignment_loss_weight_1(self, x, y, weight, alpha=2):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        loss = (x - y).norm(p=2, dim=1).pow(alpha)
        return (weight * loss).mean()

    def uniformity_loss(self, x, t=2):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def l2_reg_loss(self, reg, *args):
        emb_loss = 0
        for emb in args:
            emb_loss += torch.norm(emb, p=2)
        return emb_loss * reg

    def distance_wb(self, gwr, gws):
        shape = gwr.shape
        if len(gwr.shape) == 2:
            gwr = gwr.T
            gws = gws.T
        elif len(shape) == 4:
            gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
            gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
        elif len(shape) == 3:
            gwr = gwr.reshape(shape[0], shape[1] * shape[2])
            gws = gws.reshape(shape[0], shape[1] * shape[2])
        elif len(shape) == 1:
            gwr = gwr.reshape(1, shape[0])
            gws = gws.reshape(1, shape[0])
            return 0
        dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 1e-6))
        return dis_weight

    def match_loss(self, gw_syn, gw_real, dis_metric):
        dis = torch.tensor(0.0).to('cuda')
        if dis_metric == 'ours':
            for i in range(len(gw_real)):
                dis += self.distance_wb(gw_real[i], gw_syn[i])
        else:
            exit('Unknown distance metric')
        return dis

    def _train_epoch(self, epoch):
        start_t = time.time()

        # === Inner Loop ===
        for _ in range(self.inner_loop):
            for batch in self.dataloader:
                user_id_list, pos_item_list, neg_item_list, is_noisy_list = self.dataset.get_train_batch(batch)
                self.model.train()
                self.opt.zero_grad()

                user_emb, item_emb = self.model._get_rep()
                u_emb = user_emb[user_id_list]
                i_emb = item_emb[pos_item_list]
                j_emb = item_emb[neg_item_list]

                w_pos = self.model_generator(u_emb, i_emb).detach()
                w_neg = self.model_generator(u_emb, j_emb).detach()

                # BPR loss only used for optimization
                loss_bpr = self.bpr_loss_weight(u_emb, i_emb, j_emb, w_pos, w_neg)
                loss_align = self.alignment_loss_weight_1(u_emb, i_emb, w_pos)

                loss = (
                    loss_bpr + self.alpha * loss_align
                )
                loss.backward()
                self.opt.step()

        # === Outer Loop ===
        for _ in range(self.outer_loop):
            self.model.eval()
            user_emb, item_emb = self.model._get_rep()
            rand_user_list = np.random.randint(0, self.dataset.n_users, size=128)
            batch = self.dataset.get_train_batch(rand_user_list)
            u, i, j, _ = batch
            u_emb = user_emb[u]
            i_emb = item_emb[i]
            j_emb = item_emb[j]

            w_pos = self.model_generator(u_emb, i_emb)
            w_neg = self.model_generator(u_emb, j_emb)

            # real gradient: BPR-based
            loss_real = self.bpr_loss_weight(u_emb, i_emb, j_emb, w_pos, w_neg)
            gw_real = torch.autograd.grad(loss_real, self.model_parameters, retain_graph=True, create_graph=True)

            # synthetic gradient: AU-based
            loss_syn = self.alignment_loss_weight_1(u_emb, i_emb, w_pos)
            gw_syn = torch.autograd.grad(loss_syn, self.model_parameters, retain_graph=True, create_graph=True)

            loss_match = self.match_loss(gw_syn, gw_real, dis_metric="ours")
            loss_reg = self.l2_reg_loss(self.generator_reg, u_emb, i_emb)
            loss = self.gamma * loss_match + loss_reg

            self.generator_opt.zero_grad()
            loss.backward()
            self.generator_opt.step()

        end_t = time.time()
        print(f"Epoch {epoch}, Time: {end_t - start_t:.2f}")

class TCECFTrainer(CFTrainer):
    def __init__(self, trainer_config) -> None:
        super().__init__(trainer_config)
        
        self.device = self.config["device"]
        self.exponent = 1
        self.drop_rate = self.config["drop_rate"]
        self.num_gradual = self.config["num_gradual"]
        self.count = 0
    
    def drop_rate_schedule(self, iteration):
        drop_rate = np.linspace(0, self.drop_rate**self.exponent, self.num_gradual)
        if iteration < self.num_gradual:
            return drop_rate[iteration]
        else:
            return self.drop_rate

    def _rec_loss(self, pos_scores, neg_scores):
        # 1) 기본 BPR 형태 손실
        raw_loss = F.softplus(neg_scores - pos_scores)  # shape: (batch,)

        # 2) 현재 드롭율 t 적용
        t = self.drop_rate_schedule(self.count)
        self.count += 1

        remember_rate = 1.0 - t
        num_remember = max(int(remember_rate * raw_loss.numel()), 1)

        ind_sorted = torch.argsort(raw_loss)
        ind_update = ind_sorted[:num_remember]

        return raw_loss[ind_update]

    def _train_epoch(self, epoch):
        start_t = time.time()
        epoch_loss = 0
        
        for batch_data in self.dataloader:
            self.model.train()
            user_id_list, pos_item_list, neg_item_list, is_noisy_list = self.dataset.get_train_batch(batch_data)
            if self.config["model"] in ["NeuMF"]:
                pos_logits, neg_logits, _, _, _, l2_norm_sq = self.model(user_id_list, pos_item_list, neg_item_list)
                pos_logits = torch.sum(pos_logits, dim=1)
                neg_logits = torch.sum(neg_logits, dim=1)
            else:
                users_emb, pos_items_emb, neg_items_emb, l2_norm_sq, all_items = self.model.forward_vq(user_id_list, pos_item_list, neg_item_list)
                pos_logits = torch.sum(users_emb * pos_items_emb, dim=1)
                neg_logits = torch.sum(users_emb * neg_items_emb, dim=1)

            self.opt.zero_grad()
            loss = (self._rec_loss(pos_logits, neg_logits).mean()) + self.config["weight_decay"] * l2_norm_sq
            loss.backward()
            self.opt.step()
            epoch_loss += loss.item()

        end_t = time.time()
        print(f"Epoch {epoch}: Rec Loss: {epoch_loss/len(self.dataloader):.4f}, Time: {end_t-start_t:.2f}")
        
class RCECFTrainer(CFTrainer):
    def __init__(self, trainer_config) -> None:
        super().__init__(trainer_config)
        
        self.device = self.config["device"]
        self.beta = self.config["beta"]

    def loss_function_bpr(self, pos_scores, neg_scores):
        """
        pos_scores, neg_scores: 모델이 출력한 우선순위(logit) 벡터
        beta: 조절 파라미터 (여기선 0.2)
        """
        # 1) BPR 근사 손실: softplus(neg - pos) == log(1 + exp(neg - pos))
        raw_loss = F.softplus(neg_scores - pos_scores)      # shape (batch,)

        # 2) confidence p = sigmoid(pos - neg).detach()
        p = torch.sigmoid(pos_scores - neg_scores).detach()  # shape (batch,)

        # 3) R-CE weight
        weight = p.pow(self.beta)  # shape (batch,)

        # 4) 최종 손실
        return raw_loss * weight
       
    
    def _train_epoch(self, epoch):
        start_t = time.time()
        epoch_loss = 0
        
        for batch_data in self.dataloader:
            self.model.train()
            user_id_list, pos_item_list, neg_item_list, _ = self.dataset.get_train_batch(batch_data)
            if self.config["model"] in ["NeuMF"]:
                pos_logits, neg_logits, _, _, _, l2_norm_sq = self.model(user_id_list, pos_item_list, neg_item_list)
                pos_logits = torch.sum(pos_logits, dim=1)
                neg_logits = torch.sum(neg_logits, dim=1)
            else:
                users_emb, pos_items_emb, neg_items_emb, l2_norm_sq, all_items = self.model.forward_vq(user_id_list, pos_item_list, neg_item_list)
                pos_logits = torch.sum(users_emb * pos_items_emb, dim=1)
                neg_logits = torch.sum(users_emb * neg_items_emb, dim=1)

            self.opt.zero_grad()
            loss = (self.loss_function_bpr(pos_logits, neg_logits)).mean() + self.config["weight_decay"] * l2_norm_sq
            loss.backward()
            self.opt.step()
            epoch_loss += loss.item()
            
        end_t = time.time()
        print(f"Epoch {epoch}: Rec Loss: {epoch_loss/len(self.dataloader):.4f}, Time: {end_t-start_t:.2f}")
        
class DCFCFTrainer(CFTrainer):
    def __init__(self, config):
        super().__init__(config)

        self.device = config['device']
        self.drop_rate = config.get('drop_rate', 0.2)
        self.co_lambda = config.get('co_lambda', 0.01)
        self.relabel_ratio = config.get('relabel_ratio', 0.03)
        self.mean_loss_interval = config.get('mean_loss_interval', 2)  # ν
        self.max_relabel_epoch = config.get('max_relabel_epoch', 10)
        self.batch_size = config.get('batch_size')

        self.sn = self.dataset.get_active_train_size() if hasattr(self.dataset, "get_active_train_size") else len(self.dataset)

        # self.loss_history = []

        self.sample_loss_history = defaultdict(
            lambda: deque(maxlen=self.mean_loss_interval)
        )

        self.sample_lower_bounds = {}
        self.sample_is_relabelled = set()

    def soft_process(self, loss: torch.Tensor) -> torch.Tensor:
        return torch.log(1.0 + loss + 0.5 * loss * loss)

    def _get_hist_mean(self, sample_ids, dtype):
        """
        sample_ids: shape (batch,)
        return: shape (batch,)
        """
        hist_mean = []
        for sid in sample_ids.tolist():
            hist = self.sample_loss_history[int(sid)]
            if len(hist) == 0:
                hist_mean.append(0.0)
            else:
                hist_mean.append(sum(hist) / len(hist))
        return torch.tensor(hist_mean, device=self.device, dtype=dtype)
    
    def dcf_bpr_loss(self,
                     sample_ids: torch.Tensor,
                     pos_scores: torch.Tensor,
                     neg_scores: torch.Tensor,
                     epoch: int):
        """
        DCF correction + BPR optimization

        return:
            result["loss"]: scalar tensor
            result["damped_loss"]: shape (batch,)
            result["lower_bound"]: shape (batch,)
            result["sample_ids"]: shape (batch,)
            result["relabel_candidate"]: shape (K,)
        """
        batch_size = pos_scores.size(0)
        dtype = pos_scores.dtype

        raw_loss = F.softplus(neg_scores - pos_scores)

        damped_loss = self.soft_process(raw_loss)

        hist_mean = self._get_hist_mean(sample_ids, dtype)

        s = torch.tensor(epoch + 1.0, device=self.device, dtype=dtype)

        loss_mean = (hist_mean * s + damped_loss) / (s + 1.0)

        co_lambda = torch.tensor(self.co_lambda, device=self.device, dtype=dtype)
        sn_tensor = torch.tensor(float(self.sn), device=self.device, dtype=dtype)

        confidence_bound = (
            co_lambda * (s + (co_lambda * torch.log(2.0 * s)) / (s * s)) / ((sn_tensor + 1.0) - co_lambda)
        )

        lower_bound = F.relu(loss_mean - confidence_bound)

        inds = torch.argsort(lower_bound)

        remember_rate = 1.0 - self.drop_rate
        num_remember = max(1, int(remember_rate * batch_size))

        split = int(((1.0 - self.relabel_ratio) + self.relabel_ratio * remember_rate) * batch_size)

        split = min(max(split, 0), batch_size)

        highest_inds = inds[split:]

        saved_inds = inds[:num_remember]

        final_inds = torch.cat([highest_inds, saved_inds], dim=0).unique()

        final_pos = pos_scores[final_inds]
        final_neg = neg_scores[final_inds]
        bpr_loss = F.softplus(final_neg - final_pos).mean()

        return {
            "loss": bpr_loss,
            "damped_loss": damped_loss.detach().cpu(),
            "lower_bound": lower_bound.detach().cpu(),
            "sample_ids": sample_ids.detach().cpu(),
            "relabel_candidate": sample_ids[highest_inds].detach().cpu()
        }

    def _train_epoch_update(self,
                            sample_ids: torch.Tensor,
                            damped_loss: torch.Tensor,
                            lower_bound: torch.Tensor):
        """
        sample-wise history update
        """
        for sid, loss_val, lb_val in zip(
            sample_ids.tolist(),
            damped_loss.tolist(),
            lower_bound.tolist()
        ):
            sid = int(sid)
            self.sample_loss_history[sid].append(float(loss_val))
            self.sample_lower_bounds[sid] = float(lb_val)

    def _current_relabel_ratio(self, epoch: int):
        """
        DCF의 progressive relabel schedule
        r_i = min(i * R / O, R)
        """
        O = max(1, self.max_relabel_epoch)
        current_ratio = min(((epoch + 1) * self.relabel_ratio) / O,
                            self.relabel_ratio)
        return current_ratio

    def _apply_progressive_relabel(self, epoch: int):
        """
        BPR setting에서 relabel의 해석:
        noisy positive interaction을 다음 에폭부터 positive pool에서 제외
        """
        if len(self.sample_lower_bounds) == 0:
            return

        current_ratio = self._current_relabel_ratio(epoch)
        if current_ratio <= 0:
            return
        
        # lower bound 큰 순서 = noisy candidate
        sorted_items = [
            (sid, lb) for sid, lb in self.sample_lower_bounds.items()
            if not hasattr(self.dataset, "active_pair_mask") or self.dataset.active_pair_mask[sid]
        ]

        sorted_items = sorted(sorted_items, key=lambda x: x[1], reverse=True)

        k = int(len(sorted_items) * current_ratio)
        if k <= 0:
            return
        
        relabel_ids = [sid for sid, _ in sorted_items[:k]]

        # dataset 쪽에 구현 필요
        if hasattr(self.dataset, "mark_as_relabelled"):
            self.dataset.mark_as_relabelled(relabel_ids)

        self.sample_is_relabelled.update(relabel_ids)

        if hasattr(self.dataset, "get_active_train_size"):
            self.sn = self.dataset.get_active_train_size()

    def _train_epoch(self, epoch: int):
        start = time.time()
        total_loss = 0.0

        self.model.train()

        if hasattr(self.dataset, "resample_negatives"):
            self.dataset.resample_negatives()

        self.sample_lower_bounds = {}

        for batch_data in self.dataloader:
            # [중요] sample_ids를 batch에서 같이 받아야 함
            sample_ids, user_id_list, pos_item_list, neg_item_list, is_noisy_list = \
            self.dataset.get_train_batch_dcf(batch_data)

            if self.config['model'] == 'NeuMF':
                pos_logits, neg_logits, _, _, _, l2_norm_sq = self.model(
                    user_id_list, pos_item_list, neg_item_list
                )
                pos_scores = torch.sum(pos_logits, dim=1)
                neg_scores = torch.sum(neg_logits, dim=1)
            else:
                users_emb, pos_items_emb, neg_items_emb, l2_norm_sq, all_items = self.model.forward_vq(
                    user_id_list, pos_item_list, neg_item_list
                )
                pos_scores = torch.sum(users_emb * pos_items_emb, dim=1)
                neg_scores = torch.sum(users_emb * neg_items_emb, dim=1)

            pos_scores = pos_scores.to(self.device)
            neg_scores = neg_scores.to(self.device)
            sample_ids = sample_ids.to(self.device)

            result = self.dcf_bpr_loss(
                sample_ids,
                pos_scores,
                neg_scores,
                epoch
            )

            loss = result["loss"]

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            total_loss += loss.item()
            self._train_epoch_update(
                sample_ids=result["sample_ids"],
                damped_loss=result["damped_loss"],
                lower_bound=result["lower_bound"]
            )

        self._apply_progressive_relabel(epoch)

        elapsed = time.time() - start
        avg_loss = total_loss / len(self.dataloader)
        print(f"Epoch {epoch}: BPR Loss {avg_loss:.4f}, Time {elapsed:.2f}s")

class PRIDECFTrainer(BasicTrainer):
    """
    Warm-up: train 3 experts (O/R/T) + linear-only MoE gates (2 Linear + Softmax)
             - Expert O: vanilla BPR
             - Expert R: R-CE weighted BPR
             - Expert T: T-CE (small-loss remember)
             - Gate loss: vanilla BPR on MoE-combined embeddings (updates gate + experts)

    Transition (epoch == begin_adv, one-time):
      - Evaluate all experts fairly: per-user average BPR loss (no dropping, T-CE items included)
      - Select the expert with lowest per-user average loss
      - Switch self.model and self.opt to the selected expert (optimizer state kept)

    After begin_adv:
      - Train selected expert with PRIDE weighting (intent * cluster-stability)
      - Other experts + gate are frozen (eval mode, no optimizer step)
    """

    def __init__(self, trainer_config) -> None:
        super().__init__(trainer_config)

        # Dataset / Dataloader
        self._create_dataset(f"data/{trainer_config['dataset']}")
        self._create_dataloader()

        # Main model (PRIDE stage)
        self._create_model()
        self._create_opt()

        # Experts (warm-up stage)
        self._create_experts()
        self._create_experts_opt()

        # Device
        self.device = self.config["device"]

        # ---- PRIDE config ----
        self.num_codebook = self.config["model_config"]["denoise_config"]["num_codebook"]
        self.codebook = ResidualVQ(
            dim=self.config["out_dim"],
            codebook_size=self.num_codebook,
            num_quantizers=self.config["model_config"]["denoise_config"]["num_hirearchy"],
            decay=self.config["model_config"]["denoise_config"]["ema"],
        ).to(self.device)
        self.begin_adv = self.config["model_config"]["denoise_config"]["begin_adv"]
        self.user_interact_history = self.dataset.get_interaction_matrix(self.device)
        self.prev_centroids = []
        # PRIDE weighting config
        self.weight_mode = self.config.get("weight_mode", "noise_energy_boltzmann")
        self.energy_alpha = self.config.get("energy_alpha", 1.0)
        self.energy_beta = self.config.get("energy_beta", 1.0)
        self.energy_gamma = self.config.get("energy_gamma", 1.0)
        self.energy_r = self.config.get("energy_r", 1.0)
        self.energy_lambda = self.config.get("energy_lambda", 0.5)
        self.lambda_dis = self.config.get("lambda_dis", 1.0)
        self.tau = self.config.get("tau", 1.0)
        self.weight_eps = self.config.get("weight_eps", 1e-8)

        # ---- R-CE config ----
        self.beta = self.config["beta"]

        # ---- T-CE config ----
        self.exponent = 1
        self.drop_rate = self.config["drop_rate"]
        self.num_gradual = self.config["num_gradual"]
        self.count = 0

        # ---- MoE gating (linear-only) ----
        self._init_moe_gating()
        self._create_moe_opt()
        self._moe_initialized = False  # one-time init at epoch==begin_adv
        self.selected_expert_idx = None
        self.selected_expert_name = None

        # ---- Ablation flags ----
        self.ablation = self.config["ablation"]

    # =========================
    # Setup
    # =========================
    def _create_dataset(self, path):
        self.dataset = CFDataset(path, self.config)

    def _create_dataloader(self):
        self.dataloader = DataLoader(self.dataset, batch_size=int(self.config["batch_size"]), shuffle=True)
        print(f"Create Dataloader with batch_size:{int(self.config['batch_size'])}")

    def _create_model(self):
        glo = globals()
        self.config["model_config"]["n_users"] = self.dataset.n_users
        self.config["model_config"]["n_items"] = self.dataset.n_items
        self.model = glo[f'{self.config["model"]}'](self.config["model_config"], self.dataset)
        if torch.cuda.is_available() and self.config["use_gpu"]:
            self.model.cuda()

    def _create_opt(self):
        self.opt = optim.AdamW(self.model.parameters(), lr=self.config["lr"])

    def _create_experts(self):
        glo = globals()
        self.config["model_config"]["n_users"] = self.dataset.n_users
        self.config["model_config"]["n_items"] = self.dataset.n_items
        self.expert_o = glo[f'{self.config["model"]}'](self.config["model_config"], self.dataset)
        self.expert_r = glo[f'{self.config["model"]}'](self.config["model_config"], self.dataset)
        self.expert_t = glo[f'{self.config["model"]}'](self.config["model_config"], self.dataset)

        if torch.cuda.is_available() and self.config["use_gpu"]:
            self.expert_o.cuda()
            self.expert_r.cuda()
            self.expert_t.cuda()

        # keep in a list for clean loops
        self.experts = [self.expert_o, self.expert_r, self.expert_t]

    def _create_experts_opt(self):
        self.opt_o = optim.AdamW(self.expert_o.parameters(), lr=self.config["lr"])
        self.opt_r = optim.AdamW(self.expert_r.parameters(), lr=self.config["lr"])
        self.opt_t = optim.AdamW(self.expert_t.parameters(), lr=self.config["lr"])
        self.expert_opts = [self.opt_o, self.opt_r, self.opt_t]

    # =========================
    # MoE gating (linear-only)
    # =========================
    def _init_moe_gating(self):
        """
        Nonlinearity 없이:
          Linear -> Linear -> softmax (forward에서)
        user/item 각각 gating: input=concat(3 experts emb) => 3 weights
        """
        d = self.config["out_dim"]
        hidden = self.config.get("moe_hidden_dim", d)

        self.user_gate_fc1 = nn.Linear(d * 3, hidden).to(self.device)
        self.user_gate_fc2 = nn.Linear(hidden, 3).to(self.device)

        self.item_gate_fc1 = nn.Linear(d * 3, hidden).to(self.device)
        self.item_gate_fc2 = nn.Linear(hidden, 3).to(self.device)

    def _create_moe_opt(self):
        lr_gate = self.config.get("moe_lr", self.config["lr"])
        params = (
            list(self.user_gate_fc1.parameters())
            + list(self.user_gate_fc2.parameters())
            + list(self.item_gate_fc1.parameters())
            + list(self.item_gate_fc2.parameters())
        )
        self.opt_gate = optim.AdamW(params, lr=lr_gate)

    @torch.no_grad()
    def _softmax_gate_user(self, u_o, u_r, u_t):
        feat = torch.cat([u_o, u_r, u_t], dim=1)                 # (B, 3D)
        logits = self.user_gate_fc2(self.user_gate_fc1(feat))    # (B, 3)
        return F.softmax(logits, dim=1)

    @torch.no_grad()
    def _softmax_gate_item(self, i_o, i_r, i_t):
        feat = torch.cat([i_o, i_r, i_t], dim=1)                 # (B, 3D)
        logits = self.item_gate_fc2(self.item_gate_fc1(feat))    # (B, 3)
        return F.softmax(logits, dim=1)

    def _gate_user(self, u_o, u_r, u_t):
        feat = torch.cat([u_o, u_r, u_t], dim=1)
        logits = self.user_gate_fc2(self.user_gate_fc1(feat))
        return F.softmax(logits, dim=1)

    def _gate_item(self, i_o, i_r, i_t):
        feat = torch.cat([i_o, i_r, i_t], dim=1)
        logits = self.item_gate_fc2(self.item_gate_fc1(feat))
        return F.softmax(logits, dim=1)

    def _moe_combine(self, w, e0, e1, e2):
        """
        w: (B,3), e*: (B,D) => (B,D)
        """
        return w[:, 0:1] * e0 + w[:, 1:2] * e1 + w[:, 2:3] * e2

    # =========================
    # Loss utilities
    # =========================
    def _rec_loss(self, pos, neg):
        return F.softplus(neg - pos)

    def get_rce_weight(self, pos_scores, neg_scores):
        p = torch.sigmoid(pos_scores - neg_scores).detach()
        return p.pow(self.beta)

    def drop_rate_schedule(self, iteration):
        drop_rate = np.linspace(0, self.drop_rate ** self.exponent, self.num_gradual)
        if iteration < self.num_gradual:
            return float(drop_rate[iteration])
        return float(self.drop_rate)

    # =========================
    # Expert forward helpers
    # =========================
    def _forward_model_triplet(self, model, user_id_list, pos_item_list, neg_item_list):
        """
        Returns:
          users_emb (B,D), pos_items_emb (B,D), neg_items_emb (B,D),
          pos_logits (B,), neg_logits (B,),
          l2_norm_sq (scalar tensor),
          all_items (N,D) or None
        """
        if self.config["model"] in ["NeuMF"]:
            pos_logits, neg_logits, users_emb, pos_items_emb, neg_items_emb, l2_norm_sq = model(
                user_id_list, pos_item_list, neg_item_list
            )
            pos_logits = torch.sum(pos_logits, dim=1)
            neg_logits = torch.sum(neg_logits, dim=1)
            all_items = None
            return users_emb, pos_items_emb, neg_items_emb, pos_logits, neg_logits, l2_norm_sq, all_items

        # assume forward_vq exists for PRIDE-style code
        users_emb, pos_items_emb, neg_items_emb, l2_norm_sq, all_items = model.forward_vq(
            user_id_list, pos_item_list, neg_item_list
        )
        pos_logits = torch.sum(users_emb * pos_items_emb, dim=1)
        neg_logits = torch.sum(users_emb * neg_items_emb, dim=1)
        return users_emb, pos_items_emb, neg_items_emb, pos_logits, neg_logits, l2_norm_sq, all_items

    # =========================
    # Warm-up: train experts + gate
    # =========================
    def _loss_expert_o(self, pos_logits, neg_logits, l2_norm_sq):
        return self._rec_loss(pos_logits, neg_logits).mean() + self.config["weight_decay"] * l2_norm_sq

    def _loss_expert_r(self, pos_logits, neg_logits, l2_norm_sq):
        raw = self._rec_loss(pos_logits, neg_logits)
        w = self.get_rce_weight(pos_logits, neg_logits)
        return (raw * w).mean() + self.config["weight_decay"] * l2_norm_sq

    def _loss_expert_t(self, pos_logits, neg_logits, l2_norm_sq):
        raw = self._rec_loss(pos_logits, neg_logits)
        t = self.drop_rate_schedule(self.count)
        self.count += 1

        remember_rate = 1.0 - t
        num_remember = max(int(remember_rate * raw.numel()), 1)
        ind_sorted = torch.argsort(raw)
        ind_update = ind_sorted[:num_remember]
        return raw[ind_update].mean() + self.config["weight_decay"] * l2_norm_sq

    def _train_one_expert(self, expert, opt, loss_fn, user_id_list, pos_item_list, neg_item_list):
        expert.train()
        opt.zero_grad()

        _, _, _, pos_logits, neg_logits, l2_norm_sq, _ = self._forward_model_triplet(
            expert, user_id_list, pos_item_list, neg_item_list
        )
        loss = loss_fn(pos_logits, neg_logits, l2_norm_sq)
        loss.backward()
        opt.step()
        return float(loss.item())

    def _train_warmup_step(self, user_id_list, pos_item_list, neg_item_list):
        """
        One batch:
          1) Train each expert with its own loss (3 backward+step)
          2) Train gate (and also allow experts to receive extra gradients from MoE loss) (1 backward+step)
        """
        # ---- 1) expert updates ----
        loss_o = self._train_one_expert(self.expert_o, self.opt_o, self._loss_expert_o, user_id_list, pos_item_list, neg_item_list)
        loss_r = self._train_one_expert(self.expert_r, self.opt_r, self._loss_expert_r, user_id_list, pos_item_list, neg_item_list)
        loss_t = self._train_one_expert(self.expert_t, self.opt_t, self._loss_expert_t, user_id_list, pos_item_list, neg_item_list)

        # ---- 2) gate update (MoE BPR loss) ----
        self.expert_o.train()
        self.expert_r.train()
        self.expert_t.train()
        self.user_gate_fc1.train()
        self.user_gate_fc2.train()
        self.item_gate_fc1.train()
        self.item_gate_fc2.train()

        self.opt_gate.zero_grad()

        # forward all experts again (cheap + keeps correct graph for gate)
        u_o, pi_o, ni_o, _, _, _, _ = self._forward_model_triplet(self.expert_o, user_id_list, pos_item_list, neg_item_list)
        u_r, pi_r, ni_r, _, _, _, _ = self._forward_model_triplet(self.expert_r, user_id_list, pos_item_list, neg_item_list)
        u_t, pi_t, ni_t, _, _, _, _ = self._forward_model_triplet(self.expert_t, user_id_list, pos_item_list, neg_item_list)

        w_u = self._gate_user(u_o, u_r, u_t)          # (B,3)
        w_pi = self._gate_item(pi_o, pi_r, pi_t)      # (B,3)
        w_ni = self._gate_item(ni_o, ni_r, ni_t)      # (B,3)

        u = self._moe_combine(w_u, u_o, u_r, u_t)                 # (B,D)
        pos_i = self._moe_combine(w_pi, pi_o, pi_r, pi_t)         # (B,D)
        neg_i = self._moe_combine(w_ni, ni_o, ni_r, ni_t)         # (B,D)

        pos_logits = torch.sum(u * pos_i, dim=1)
        neg_logits = torch.sum(u * neg_i, dim=1)

        # gate trains toward minimizing vanilla BPR on combined representation
        gate_loss = self._rec_loss(pos_logits, neg_logits).mean()
        gate_loss.backward()
        self.opt_gate.step()

        return loss_o, loss_r, loss_t, float(gate_loss.item())

    # =========================
    # MoE init -> overwrite main model embeddings
    # =========================
    def _get_embedding_param(self, model, kind: str):
        """
        kind in {"user","item"}.
        Tries common attribute names used in MF/LightGCN/others.
        """
        candidates_user = ["embedding_user", "user_embedding", "user_emb", "users_emb", "user_embeds", "user_embedding_table"]
        candidates_item = ["embedding_item", "item_embedding", "item_emb", "items_emb", "item_embeds", "item_embedding_table"]

        names = candidates_user if kind == "user" else candidates_item
        for name in names:
            if hasattr(model, name):
                obj = getattr(model, name)
                if isinstance(obj, nn.Embedding):
                    return obj.weight
                if torch.is_tensor(obj) and obj.requires_grad:
                    return obj
                if hasattr(obj, "weight") and torch.is_tensor(obj.weight):
                    return obj.weight
        raise AttributeError(f"[MoE init] Cannot find {kind} embedding parameter in model. Add a getter in your model.")

    @torch.no_grad()
    def _get_all_user_emb(self, model):
        # prefer explicit getter if exists
        if hasattr(model, "get_all_user_emb"):
            return model.get_all_user_emb()
        # fallback to embedding parameter
        w = self._get_embedding_param(model, "user")
        return w.detach()

    @torch.no_grad()
    def _get_all_item_emb(self, model):
        if hasattr(model, "get_all_item_emb"):
            return model.get_all_item_emb()
        w = self._get_embedding_param(model, "item")
        return w.detach()

    @torch.no_grad()
    def _overwrite_main_embeddings_from_moe(self):
        """
        Build MoE-combined embedding tables and copy into self.model user/item embeddings.
        Uses gates computed from (u_o,u_r,u_t) and (i_o,i_r,i_t) only (per-entity gating).
        """
        self.user_gate_fc1.eval()
        self.user_gate_fc2.eval()
        self.item_gate_fc1.eval()
        self.item_gate_fc2.eval()

        u_o = self._get_all_user_emb(self.expert_o).to(self.device)
        u_r = self._get_all_user_emb(self.expert_r).to(self.device)
        u_t = self._get_all_user_emb(self.expert_t).to(self.device)

        i_o = self._get_all_item_emb(self.expert_o).to(self.device)
        i_r = self._get_all_item_emb(self.expert_r).to(self.device)
        i_t = self._get_all_item_emb(self.expert_t).to(self.device)

        # user gate: (U,3)
        wu = self._softmax_gate_user(u_o, u_r, u_t)

        U = wu.size(0)
        sample_u = torch.randperm(U, device=wu.device)[:min(20, U)]
        self._log_gate_stats("user", wu, sample_idx=sample_u)

        u = wu[:, 0:1] * u_o + wu[:, 1:2] * u_r + wu[:, 2:3] * u_t

        # item gate: (I,3)
        wi = self._softmax_gate_item(i_o, i_r, i_t)

        I = wi.size(0)
        sample_i = torch.randperm(I, device=wi.device)[:min(20, I)]
        self._log_gate_stats("item", wi, sample_idx=sample_i)

        it = wi[:, 0:1] * i_o + wi[:, 1:2] * i_r + wi[:, 2:3] * i_t

        # ✅ (옵션) gate 테이블 저장 (U×3, I×3)
        if self.config.get("save_gate_tables", True):
            base_dir = f"{self.config['checkpoints']}/{self.config['model']}/{self.config['method']}/{self.config['dataset']}"
            if self.config.get("main_file", "") != "":
                base_dir = os.path.join(base_dir, self.config["main_file"])
            os.makedirs(base_dir, exist_ok=True)

            save_path = os.path.join(
                base_dir,
                f"gate_tables_seed{self.config['seed']}_begin{self.begin_adv}.pt"
            )
            torch.save(
                {"wu": wu.detach().cpu(), "wi": wi.detach().cpu()},
                save_path
            )
            print(f"[GateSaved] {save_path}")

        # overwrite main model embedding tables
        user_param = self._get_embedding_param(self.model, "user")
        item_param = self._get_embedding_param(self.model, "item")

        if user_param.shape != u.shape or item_param.shape != it.shape:
            raise ValueError(f"[MoE init] Shape mismatch: main_user={tuple(user_param.shape)}, moe_user={tuple(u.shape)}, "
                             f"main_item={tuple(item_param.shape)}, moe_item={tuple(it.shape)}")

        user_param.copy_(u)
        item_param.copy_(it)

    # =========================
    # PRIDE phase helpers
    # =========================
    @torch.no_grad()
    def save_previous_codebooks(self):
        # Save BEFORE EMA update so curr-prev reflects actual epoch-to-epoch drift
        self.prev_centroids = []
        for layer in self.codebook.layers:
            self.prev_centroids.append(layer.codebook.clone().detach())
        all_items = self.model.get_all_item_emb()
        _ = self.codebook(all_items)  # EMA update with all items (advances codebook to "curr")
        self._log_codebook_reinit()

    def _validate_pride_config(self):
        valid_modes = {
            "noise_energy_boltzmann",
            "reliability_boltzmann",
            "disagreement_aware",
            "power_product",
            "lambda_power",
        }
        valid_ablations = {"full", "wo_pride", "wo_user_intent", "wo_stability"}

        if self.weight_mode not in valid_modes:
            raise ValueError(f"Invalid weight_mode: {self.weight_mode}. Expected one of {sorted(valid_modes)}")
        if self.ablation not in valid_ablations:
            raise ValueError(f"Invalid ablation option: {self.ablation}. Expected one of {sorted(valid_ablations)}")
        if self.tau <= 0:
            raise ValueError(f"tau must be > 0, got {self.tau}")
        if self.weight_eps <= 0:
            raise ValueError(f"weight_eps must be > 0, got {self.weight_eps}")

    def _compute_pride_signals(self, all_items, user_id_list, pos_item_list, pos_items_emb):
        # Use codebook in eval mode: get assignments without triggering another EMA update.
        # self.codebook has already been advanced by save_previous_codebooks (curr state).
        was_training = self.codebook.training
        self.codebook.eval()
        with torch.no_grad():
            _, quantized_idx, _ = self.codebook(all_items)  # idx: [num_hirec, N]
        if was_training:
            self.codebook.train()

        first_layer_idx = quantized_idx.T[0]  # [N]
        item_cluster_onehot = F.one_hot(first_layer_idx, num_classes=self.num_codebook).float()  # [N,C]

        user_ids = torch.tensor(user_id_list, device=self.device)
        user_hist_all = torch.sparse.mm(self.user_interact_history, item_cluster_onehot)  # [U,C]
        user_hist = user_hist_all[user_ids].to(self.device)  # [B,C]
        user_attn = user_hist / (user_hist.sum(dim=1, keepdim=True) + 1e-8)  # [B,C]

        codebook_vecs = self.codebook.layers[0].codebook  # [C,D]
        user_centroid = torch.matmul(user_attn, codebook_vecs)  # [B,D]

        user_centroid = F.normalize(user_centroid, dim=1)
        pos_vec = F.normalize(pos_items_emb, dim=1)
        cosine_sim = torch.sum(user_centroid * pos_vec, dim=1)  # [B]
        cosine_sim = (cosine_sim + 1.0) / 2.0  # [0,1]
        c_tilde = cosine_sim.clamp(0.0, 1.0)

        prev = self.prev_centroids[0]
        curr = self.codebook.layers[0].codebook.detach()  # post-EMA state (advanced by save_previous_codebooks)

        delta = torch.norm(curr - prev, dim=1)
        norm_delta = (delta - delta.min()) / (delta.max() - delta.min() + 1e-8)
        inv_delta = 1.0 - norm_delta  # stability

        pos_code_idx = quantized_idx.T[0][pos_item_list]
        cluster_stability = inv_delta[pos_code_idx]  # [B]
        s_tilde = cluster_stability.clamp(0.0, 1.0)
        return s_tilde, c_tilde

    def _build_pride_terms(self, s_tilde, c_tilde):
        alpha = self.energy_alpha
        beta = self.energy_beta

        if self.ablation == "wo_user_intent":
            return {
                "s_term": alpha * (1.0 - s_tilde),
                "c_term": torch.zeros_like(c_tilde),
                "reliability": alpha * s_tilde,
                "disagreement": torch.zeros_like(c_tilde),
            }
        if self.ablation == "wo_stability":
            return {
                "s_term": torch.zeros_like(s_tilde),
                "c_term": beta * (1.0 - c_tilde),
                "reliability": beta * c_tilde,
                "disagreement": torch.zeros_like(c_tilde),
            }
        if self.ablation == "full":
            return {
                "s_term": alpha * (1.0 - s_tilde),
                "c_term": beta * (1.0 - c_tilde),
                "reliability": alpha * s_tilde + beta * c_tilde,
                "disagreement": self.lambda_dis * torch.abs(s_tilde - c_tilde),
            }
        raise ValueError(f"Invalid ablation option: {self.ablation}")

    def _compute_pride_weight(self, s_tilde, c_tilde, return_components=False):
        if self.ablation == "wo_pride":
            w = torch.ones_like(c_tilde)
            if return_components:
                return w, {}
            return w

        gamma = self.energy_gamma
        eps = self.weight_eps
        terms = self._build_pride_terms(s_tilde, c_tilde)

        if self.weight_mode == "noise_energy_boltzmann":
            energy = terms["s_term"] + terms["c_term"]
            return torch.exp(-gamma * energy)

        if self.weight_mode == "reliability_boltzmann":
            logits = (terms["reliability"] / self.tau).clamp(max=50.0)
            return torch.exp(logits)

        if self.weight_mode == "disagreement_aware":
            energy = terms["s_term"] + terms["c_term"] + terms["disagreement"]
            return torch.exp(-gamma * energy)

        if self.weight_mode == "power_product":
            if self.ablation == "wo_user_intent":
                w_raw = (s_tilde + eps).pow(self.energy_alpha)
            elif self.ablation == "wo_stability":
                w_raw = (c_tilde + eps).pow(self.energy_beta)
            elif self.ablation == "full":
                w_raw = (s_tilde + eps).pow(self.energy_alpha) * (c_tilde + eps).pow(self.energy_beta)
            else:
                raise ValueError(f"Invalid ablation option: {self.ablation}")

            w_min = w_raw.min()
            w_max = w_raw.max()
            if (w_max - w_min) < eps:
                return torch.ones_like(w_raw)
            return (w_raw - w_min) / (w_max - w_min + eps)

        if self.weight_mode == "lambda_power":
            # w_raw = (s + eps)^{r(1-λ)} * (c + eps)^{r·λ}
            # r: sharpness (higher → more focus on high-confidence samples)
            # λ: how much to emphasize intent(c) over stability(s)
            lam = self.energy_lambda
            r = self.energy_r
            eps = self.weight_eps
            if self.ablation == "wo_user_intent":
                s_pow = (s_tilde + eps).pow(r)
                c_pow = torch.ones_like(s_tilde)
            elif self.ablation == "wo_stability":
                s_pow = torch.ones_like(c_tilde)
                c_pow = (c_tilde + eps).pow(r)
            else:
                s_pow = (s_tilde + eps).pow(r * (1.0 - lam))
                c_pow = (c_tilde + eps).pow(r * lam)
            w_raw = s_pow * c_pow
            w_min, w_max = w_raw.min(), w_raw.max()
            if (w_max - w_min) < eps:
                w = torch.ones_like(w_raw)
            else:
                w = (w_raw - w_min) / (w_max - w_min + eps)
            if return_components:
                return w, {
                    "s_pow": s_pow.detach(),
                    "c_pow": c_pow.detach(),
                    "w_raw": w_raw.detach(),
                }
            return w

        raise ValueError(f"Invalid weight_mode: {self.weight_mode}")

    def _pride_weights(self, all_items, user_id_list, pos_item_list, pos_items_emb):
        """
        Compute PRIDE weight for each (u, pos_i).
        """
        # safety: first PRIDE step needs prev_centroids
        if len(self.prev_centroids) == 0:
            self.save_previous_codebooks()
        self._validate_pride_config()
        s_tilde, c_tilde = self._compute_pride_signals(all_items, user_id_list, pos_item_list, pos_items_emb)
        w = self._compute_pride_weight(s_tilde, c_tilde)
        return w.detach()

    def _train_pride_step(self, user_id_list, pos_item_list, neg_item_list):
        self.model.train()
        self.opt.zero_grad()

        u, pi, ni, pos_logits, neg_logits, l2_norm_sq, all_items = self._forward_model_triplet(
            self.model, user_id_list, pos_item_list, neg_item_list
        )

        # PRIDE weights
        if all_items is None:
            # NeuMF path: need item embedding table for VQ (fallback)
            all_items = self.model.get_all_item_emb()

        if len(self.prev_centroids) == 0:
            self.save_previous_codebooks()
        self._validate_pride_config()
        s_tilde, c_tilde = self._compute_pride_signals(all_items, user_id_list, pos_item_list, pi)
        if self.weight_mode == "lambda_power":
            w, comps = self._compute_pride_weight(s_tilde, c_tilde, return_components=True)
            w = w.detach()
            comps = {k: v.cpu() for k, v in comps.items()}
        else:
            w = self._compute_pride_weight(s_tilde, c_tilde).detach()
            comps = None

        loss = (self._rec_loss(pos_logits, neg_logits) * w).mean() + self.config["weight_decay"] * l2_norm_sq

        loss.backward()
        self.opt.step()
        return (
            float(loss.item()),
            s_tilde.detach().cpu(),
            c_tilde.detach().cpu(),
            w.detach().cpu(),
            comps,
        )

    # =========================
    # Training epoch
    # =========================
    def _train_epoch(self, epoch):
        start_t = time.time()

        sum_o = 0.0
        sum_r = 0.0
        sum_t = 0.0
        sum_gate = 0.0
        sum_rq = 0.0

        # buffers for PRIDE signal logging
        _s_buf, _c_buf, _w_buf, _noisy_buf, _comp_bufs = [], [], [], [], []

        for batch_data in self.dataloader:
            user_id_list, pos_item_list, neg_item_list, is_noisy = self.dataset.get_train_batch(batch_data)

            # ---- Warm-up: experts + gate ----
            if epoch < self.begin_adv:
                lo, lr, lt, lg = self._train_warmup_step(user_id_list, pos_item_list, neg_item_list)
                sum_o += lo; sum_r += lr; sum_t += lt; sum_gate += lg
                continue

            # ---- Transition: one-time expert selection + switch (optimizer state 유지) ----
            if (epoch >= self.begin_adv) and (not self._moe_initialized):
                self._select_expert_fair()
                self._switch_to_expert()
                self.save_previous_codebooks()
                self._moe_initialized = True

            # ---- PRIDE: selected expert ----
            l, s_cpu, c_cpu, w_cpu, comps_cpu = self._train_pride_step(user_id_list, pos_item_list, neg_item_list)
            sum_rq += l
            _s_buf.append(s_cpu)
            _c_buf.append(c_cpu)
            _w_buf.append(w_cpu)
            _noisy_buf.append(torch.tensor(is_noisy, dtype=torch.uint8))
            if comps_cpu is not None:
                _comp_bufs.append(comps_cpu)

        if epoch >= self.begin_adv - 1 and self._moe_initialized:
            self.save_previous_codebooks()

        end_t = time.time()
        n = max(len(self.dataloader), 1)

        if epoch < self.begin_adv:
            print(
                f"[Warm-up] Epoch {epoch}: "
                f"O={sum_o/n:.4f}, R={sum_r/n:.4f}, T={sum_t/n:.4f}, GateLoss={sum_gate/n:.4f}, "
                f"Time={end_t-start_t:.2f}"
            )
            self._log_gate_line(epoch)
        else:
            print(
                f"[PRIDE] Epoch {epoch}: "
                f"Loss={sum_rq/n:.4f}, Time={end_t-start_t:.2f}"
            )
            if (epoch + 1) % self.config.get("val_interval", 1) == 0 and _s_buf:
                comps_all = (
                    {k: torch.cat([d[k] for d in _comp_bufs]) for k in _comp_bufs[0]}
                    if _comp_bufs else None
                )
                self._save_pride_signals(
                    epoch,
                    torch.cat(_s_buf),
                    torch.cat(_c_buf),
                    torch.cat(_w_buf),
                    torch.cat(_noisy_buf),
                    comps_all,
                )

    # =========================
    # Eval
    # =========================
    def _eval_model(self, epoch=0, eval_type="val", model=None):
        """model=None이면 self.model 사용. 개별 expert 평가 시 model 인자로 전달."""
        start_t = time.time()
        assert eval_type in ["val", "test"]
        m = model if model is not None else self.model
        m.eval()

        top_ks = self.config["rec_top_k"]
        recall_list = [0.0 for _ in top_ks]
        ndcg_list = [0.0 for _ in top_ks]

        user_list = list(range(self.dataset.n_users))
        for batch_data in batch_split(users=user_list, batch_size=self.config["test_batch_size"]):
            if eval_type == "val":
                user_id_list, user_inter_list, user_train_list = self.dataset.get_val_batch(batch_data)
            else:
                user_id_list, user_inter_list, user_train_list = self.dataset.get_test_batch(batch_data)

            with torch.no_grad():
                score_list = m.predict(user_id_list).to(self.device)  # (B, num_items)

            # mask train items
            for idx, user_train_items in enumerate(user_train_list):
                if len(user_train_items) > 0:
                    train_items_tensor = torch.tensor(user_train_items, dtype=torch.long, device=self.device)
                    score_list[idx].index_fill_(0, train_items_tensor, float("-inf"))

            max_k = max(top_ks)
            for user_idx, user_inter_items in enumerate(user_inter_list):
                gt_set = set(user_inter_items)
                _, top_indices = torch.topk(score_list[user_idx], max_k)
                top_indices = top_indices.tolist()

                for j, k in enumerate(top_ks):
                    top_k = top_indices[:k]
                    num_hits = sum([1 for it in top_k if it in gt_set])
                    recall_k = num_hits / len(gt_set) if gt_set else 0.0

                    dcg = sum([1 / np.log2(i + 2) for i, it in enumerate(top_k) if it in gt_set])
                    idcg = sum([1.0 / np.log2(i + 2) for i in range(len(gt_set))])
                    ndcg_k = dcg / idcg if idcg > 0 else 0.0

                    recall_list[j] += recall_k
                    ndcg_list[j] += ndcg_k

        avg_hr = [hr / self.dataset.n_users for hr in recall_list]
        avg_ndcg = [ndcg / self.dataset.n_users for ndcg in ndcg_list]

        end_t = time.time()
        print(("Validation - " if eval_type == "val" else "Test - ") + f"Time: {end_t - start_t:.2f}")

        epoch_text = f"at Epoch {epoch}" if eval_type == "val" else ""
        self._print_performance(
            "Recommendation Performance" + epoch_text,
            ("Recall", "NDCG"),
            avg_hr,
            avg_ndcg,
            self.config["rec_top_k"],
            eval_type=eval_type,
        )
        return recall_list, ndcg_list

    def _print_performance(self, title, metrics, m1_list, m2_list, top_k_list, eval_type):
        out_text = f"{title}:"
        for i, k in enumerate(top_k_list):
            out_text += f"\n{metrics[0]}@{k}: {m1_list[i]:.4f}, {metrics[1]}@{k}: {m2_list[i]:.4f};"
            if eval_type == "val":
                self.monitor.log({f"valid_{metrics[0]}@{k}": m1_list[i], f"valid_{metrics[1]}@{k}": m2_list[i]})
            else:
                self.monitor.log({f"test_{metrics[0]}@{k}": m1_list[i], f"test_{metrics[1]}@{k}": m2_list[i]})
        print(out_text)

    @torch.no_grad()
    def _eval_moe(self, epoch=0, eval_type="val"):
        start_t = time.time()
        assert eval_type in ["val", "test"]

        # set eval mode
        self.expert_o.eval(); self.expert_r.eval(); self.expert_t.eval()
        self.user_gate_fc1.eval(); self.user_gate_fc2.eval()
        self.item_gate_fc1.eval(); self.item_gate_fc2.eval()

        top_ks = self.config["rec_top_k"]
        recall_list = [0.0 for _ in top_ks]
        ndcg_list = [0.0 for _ in top_ks]

        # ---- precompute item-gate weights for all items (I,3) ----
        # item embeddings from each expert
        i_o = self._get_all_item_emb(self.expert_o).to(self.device)   # (I,D)
        i_r = self._get_all_item_emb(self.expert_r).to(self.device)
        i_t = self._get_all_item_emb(self.expert_t).to(self.device)

        w_item = self._softmax_gate_item(i_o, i_r, i_t)               # (I,3)

        user_list = list(range(self.dataset.n_users))
        for batch_data in batch_split(users=user_list, batch_size=self.config["test_batch_size"]):

            if eval_type == "val":
                user_id_list, user_inter_list, user_train_list = self.dataset.get_val_batch(batch_data)
            else:
                user_id_list, user_inter_list, user_train_list = self.dataset.get_test_batch(batch_data)

            user_id_tensor = torch.tensor(user_id_list, device=self.device, dtype=torch.long)

            # ---- expert scores (B,I) ----
            s_o = self.expert_o.predict(user_id_list).to(self.device)
            s_r = self.expert_r.predict(user_id_list).to(self.device)
            s_t = self.expert_t.predict(user_id_list).to(self.device)

            # ---- user-gate weights (B,3) ----
            u_o = self._get_all_user_emb(self.expert_o)[user_id_tensor].to(self.device)  # (B,D)
            u_r = self._get_all_user_emb(self.expert_r)[user_id_tensor].to(self.device)
            u_t = self._get_all_user_emb(self.expert_t)[user_id_tensor].to(self.device)
            w_user = self._softmax_gate_user(u_o, u_r, u_t)                                # (B,3)

            # ---- combine scores: S = sum_k w_user[b,k] * w_item[i,k] * S_k[b,i] ----
            # (B,3) @ (3,I) 형태로 만들기 위해 per-expert weight product를 브로드캐스팅
            w0 = (w_user[:, 0:1] * w_item[:, 0].unsqueeze(0))  # (B,I)
            w1 = (w_user[:, 1:2] * w_item[:, 1].unsqueeze(0))
            w2 = (w_user[:, 2:3] * w_item[:, 2].unsqueeze(0))
            score_list = w0 * s_o + w1 * s_r + w2 * s_t        # (B,I)

            # ---- mask train items ----
            for idx, user_train_items in enumerate(user_train_list):
                if len(user_train_items) > 0:
                    train_items_tensor = torch.tensor(user_train_items, dtype=torch.long, device=self.device)
                    score_list[idx].index_fill_(0, train_items_tensor, float("-inf"))

            max_k = max(top_ks)
            for user_idx, user_inter_items in enumerate(user_inter_list):
                gt_set = set(user_inter_items)
                _, top_indices = torch.topk(score_list[user_idx], max_k)
                top_indices = top_indices.tolist()

                for j, k in enumerate(top_ks):
                    top_k = top_indices[:k]
                    num_hits = sum([1 for it in top_k if it in gt_set])
                    recall_k = num_hits / len(gt_set) if gt_set else 0.0

                    dcg = sum([1 / np.log2(i + 2) for i, it in enumerate(top_k) if it in gt_set])
                    idcg = sum([1.0 / np.log2(i + 2) for i in range(len(gt_set))])
                    ndcg_k = dcg / idcg if idcg > 0 else 0.0

                    recall_list[j] += recall_k
                    ndcg_list[j] += ndcg_k

        avg_hr = [hr / self.dataset.n_users for hr in recall_list]
        avg_ndcg = [ndcg / self.dataset.n_users for ndcg in ndcg_list]

        end_t = time.time()
        print(f"[MoE Eval] ({eval_type}) epoch={epoch} time={end_t-start_t:.2f}")
        self._print_performance(
            f"MoE Recommendation Performance at Epoch {epoch}" if eval_type == "val" else "MoE Recommendation Performance",
            ("Recall", "NDCG"),
            avg_hr,
            avg_ndcg,
            top_ks,
            eval_type=eval_type
        )
        return recall_list, ndcg_list
    
    def train(self, path=None):
        patience = self.config["patience"]
        best_metrics = -1
        _expert_names = ["O(BPR)", "R(R-CE)", "T(T-CE)"]

        best_model_path = f"{self.config['checkpoints']}/{self.config['model']}/{self.config['method']}/{self.config['dataset']}"
        if self.config["main_file"] != "":
            best_model_path = os.path.join(best_model_path, self.config["main_file"])
        if path is not None:
            best_model_path = path
        os.makedirs(best_model_path, exist_ok=True)
        best_model_path = os.path.join(best_model_path, f"{self.config['noise']}_{self.config['seed']}.pth")

        for epoch in range(self.n_epochs):
            self._train_epoch(epoch)

            if (epoch + 1) % self.config["val_interval"] == 0:
                if epoch < self.begin_adv:
                    # MoE combined eval
                    self._eval_moe(epoch, eval_type="val")
                    # Individual expert eval → [Expert Val@20]
                    expert_recall_scores = []
                    for expert in self.experts:
                        recall_list, _ = self._eval_model(epoch, eval_type="val", model=expert)
                        expert_recall_scores.append(recall_list[0])
                    scores_str = "  ".join(
                        f"{name}={s:.4f}" for name, s in zip(_expert_names, expert_recall_scores)
                    )
                    print(f"[Expert Val@20] epoch={epoch}: {scores_str}")
                else:
                    metrics_list, _ = self._eval_model(epoch, eval_type="val")
                    metrics = metrics_list[0]
                    if (epoch + 1) >= self.config["min_epochs"]:
                        if metrics > best_metrics:
                            best_metrics = metrics
                            self._save_model(best_model_path)
                            patience = self.config["patience"]
                        else:
                            patience -= 1
                            if patience <= 0:
                                print("Early stopping!")
                                break

        self._load_model(best_model_path)
        avg_hr, avg_ndcg = self._eval_model(eval_type="test")
        return avg_hr, avg_ndcg
    
    @torch.no_grad()
    def _gate_entropy(self, w: torch.Tensor, eps: float = 1e-12):
        """
        w: (N,3) softmax prob
        return: (N,) entropy
        """
        w = torch.clamp(w, min=eps, max=1.0)
        return -(w * torch.log(w)).sum(dim=1)
    
    @torch.no_grad()
    def _log_gate_stats(self, name: str, w: torch.Tensor, sample_idx: torch.Tensor = None):
        """
        name: 'user' or 'item'
        w: (N,3)
        sample_idx: optional (M,) indices to print/inspect
        """
        # basic stats
        mean = w.mean(dim=0).detach().cpu().numpy()
        std  = w.std(dim=0, unbiased=False).detach().cpu().numpy()

        argm = torch.argmax(w, dim=1)
        frac = torch.stack([(argm == k).float().mean() for k in range(3)]).detach().cpu().numpy()

        ent = self._gate_entropy(w).mean().detach().cpu().item()

        # print
        print(f"[GateStats:{name}] mean={mean}, std={std}, argmax_frac={frac}, entropy_mean={ent:.4f}")

        # wandb/monitor log (가능하면)
        if hasattr(self, "monitor") and self.monitor is not None:
            self.monitor.log({
                f"gate_{name}_w0_mean": float(mean[0]),
                f"gate_{name}_w1_mean": float(mean[1]),
                f"gate_{name}_w2_mean": float(mean[2]),
                f"gate_{name}_w0_std": float(std[0]),
                f"gate_{name}_w1_std": float(std[1]),
                f"gate_{name}_w2_std": float(std[2]),
                f"gate_{name}_argmax0_frac": float(frac[0]),
                f"gate_{name}_argmax1_frac": float(frac[1]),
                f"gate_{name}_argmax2_frac": float(frac[2]),
                f"gate_{name}_entropy_mean": float(ent),
            })

        # optional samples
        if sample_idx is not None:
            sample_idx = sample_idx.detach().cpu().numpy().tolist()
            ws = w[sample_idx].detach().cpu().numpy()
            m = min(len(sample_idx), 10)
            print(f"[GateSamples:{name}] idx[:{m}]={sample_idx[:m]}")
            print(f"[GateSamples:{name}] w[:{m}]={ws[:m]}")

    # =========================
    # Expert selection (fair, per-user average)
    # =========================
    @torch.no_grad()
    def _select_expert_fair(self):
        """
        유저별 평균 BPR loss 기반으로 expert 선택.
        - 모든 expert를 eval 모드로 실행 (dropping 없음)
        - 각 유저의 pos 아이템 전체에 대해 loss 계산 후 유저 내 평균
        - 전체 유저 평균 (argmin) → T-CE의 drop 편향 제거
        """
        import random as _random
        U = self.dataset.n_users
        I = self.dataset.n_items
        _names = ["expert_o(BPR)", "expert_r(R-CE)", "expert_t(T-CE)"]
        batch_size = int(self.config.get("selector_batch_size", 4096))

        for expert in self.experts:
            expert.eval()

        # 유저별 (pos, neg) 쌍 구성 — 삽입 순서 유지 (Python 3.7+)
        user_pairs = {}
        for user in range(U):
            pos_items = self.dataset.train_data[user]
            if not pos_items:
                continue
            pos_set = set(pos_items)
            pairs = []
            for pos in pos_items:
                neg = _random.randint(0, I - 1)
                while neg in pos_set:
                    neg = _random.randint(0, I - 1)
                pairs.append((pos, neg))
            user_pairs[user] = pairs

        # 전체 쌍 flatten (유저 순서 유지)
        all_users, all_pos, all_neg = [], [], []
        for user, pairs in user_pairs.items():
            for pos, neg in pairs:
                all_users.append(user)
                all_pos.append(pos)
                all_neg.append(neg)
        n_total = len(all_users)

        user_avg_losses = []
        for k, expert in enumerate(self.experts):
            # 배치 단위로 loss 계산
            losses_flat = []
            for start in range(0, n_total, batch_size):
                end = min(start + batch_size, n_total)
                _, _, _, pos_logits, neg_logits, _, _ = self._forward_model_triplet(
                    expert,
                    all_users[start:end],
                    all_pos[start:end],
                    all_neg[start:end],
                )
                losses_flat.extend(self._rec_loss(pos_logits, neg_logits).cpu().tolist())

            # 유저별 평균 → 전체 유저 평균
            idx = 0
            per_user_avgs = []
            for user, pairs in user_pairs.items():
                n = len(pairs)
                per_user_avgs.append(sum(losses_flat[idx: idx + n]) / n)
                idx += n

            avg = sum(per_user_avgs) / max(len(per_user_avgs), 1)
            user_avg_losses.append(avg)

        k = int(np.argmin(user_avg_losses))
        losses_str = "  ".join(
            f"{name}={l:.4f}" for name, l in zip(_names, user_avg_losses)
        )
        print(f"[MoE Select] user_avg_loss: {losses_str}")
        print(f"[MoE Select] -> {k} ({_names[k]})")
        print(f"[MoE Select] PRIDE will continue on {_names[k]} (optimizer state kept)")

        self.selected_expert_idx = k
        self.selected_expert_name = _names[k]

    def _switch_to_expert(self):
        """선택된 expert를 self.model / self.opt로 연결 (optimizer state 유지)."""
        k = self.selected_expert_idx
        self.model = self.experts[k]
        self.opt = self.expert_opts[k]

        for expert in self.experts:
            expert.eval()
        self.user_gate_fc1.eval(); self.user_gate_fc2.eval()
        self.item_gate_fc1.eval(); self.item_gate_fc2.eval()

    @torch.no_grad()
    def _log_gate_line(self, epoch):
        """warm-up 에폭 끝에 user gate 평균 가중치를 출력한다."""
        self.user_gate_fc1.eval(); self.user_gate_fc2.eval()

        u_o = self._get_all_user_emb(self.expert_o).to(self.device)
        u_r = self._get_all_user_emb(self.expert_r).to(self.device)
        u_t = self._get_all_user_emb(self.expert_t).to(self.device)

        w = self._softmax_gate_user(u_o, u_r, u_t)  # (U, 3)
        mean_w = w.mean(dim=0).cpu()

        print(
            f"[Gate]    Epoch {epoch}: "
            f"SoftTarget(user) O={mean_w[0]:.3f} R={mean_w[1]:.3f} T={mean_w[2]:.3f} | "
            f"UserGatePred     O={mean_w[0]:.3f} R={mean_w[1]:.3f} T={mean_w[2]:.3f}"
        )

    def _save_pride_signals(self, epoch, s_all, c_all, w_all, noisy_all, comps_all=None):
        """
        에폭 단위로 s_tilde / c_tilde / w / is_noisy 를 디스크에 저장.
        lambda_power mode일 때는 comps_all(s_pow, c_pow, w_raw)도 함께 저장.
        저장 경로: analyze/requiem_signals/{dataset}/noise{noise}/seed{seed}/epoch{epoch:03d}.pt
        """
        save_dir = os.path.join(
            "analyze", "requiem_signals",
            str(self.config["dataset"]),
            f"noise{self.config['noise']}",
            f"seed{self.config['seed']}",
        )
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"epoch{epoch:03d}.pt")
        payload = {
            "epoch":       epoch,
            "noise_ratio": self.config["noise"],
            "s_tilde":     s_all,      # (N,) float32
            "c_tilde":     c_all,      # (N,) float32
            "w":           w_all,      # (N,) float32
            "is_noisy":    noisy_all,  # (N,) uint8  0=clean, 1=noisy
        }
        if comps_all is not None:
            payload.update(comps_all)  # s_pow, c_pow, w_raw  (N,) float32
        torch.save(payload, save_path)
        print(f"[SignalSaved] {save_path}  (N={s_all.shape[0]})")

