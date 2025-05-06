import json
import os
import time
import numpy as np
import torch
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
from model.BOD.GraphGenerator_VAE import GraphGenerator_VAE

class BasicTrainer:
    def __init__(self, trainer_config) -> None:
        self.config = trainer_config
        self.device = trainer_config['device']
        self.n_epochs = trainer_config['n_epochs']
        self.min_epochs = trainer_config['min_epochs']
        self.max_patience = trainer_config.get('patience', 50)
        self.val_interval = trainer_config.get('val_interval', 1)
    
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
                metrics_list, _ = self._eval_model(epoch)
                metrics = metrics_list[0]
                if (epoch + 1) >= self.config["min_epochs"]:
                    if metrics > best_metrics:
                        best_metrics = metrics
                        # Save the best model
                        self._save_model(best_model_path)
                        patience = self.config["patience"]
                    else:
                        patience -= self.config["val_interval"]
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
            user_id_list, pos_item_list, neg_item_list = self.dataset.get_train_batch(batch_data)
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
        self._print_performance("Recommendation Performance" + epoch_text, ("Recall", "NDCG"), avg_hr, avg_ndcg, self.config["rec_top_k"])

        return recall_list, ndcg_list


    
    def _print_performance(self, title, metrics, m1_list, m2_list, top_k_list):
        out_text = f"{title}:"
        for i, k in enumerate(top_k_list):
            out_text += f"\n{metrics[0]}@{k}: {m1_list[i]:.4f}, {metrics[1]}@{k}: {m2_list[i]:.4f};"
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
                user_id_list, pos_item_list, neg_item_list = self.dataset.get_train_batch(batch_data, multi_sample=True, k=self.config["model_config"]['denoise_config']["item_num"])
                
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
                user_id_list, pos_item_list, neg_item_list = self.dataset.get_train_batch(batch_data)
            
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


class VQCFTrainer(CFTrainer):
    def __init__(self, trainer_config) -> None:
        super().__init__(trainer_config)
        self.num_codebook = self.config["model_config"]['denoise_config']['num_codebook']
        self.device = self.config["device"]
        self.codebook = ResidualVQ(
                dim = self.config['out_dim'],
                codebook_size = self.config["model_config"]['denoise_config']['num_codebook'],
                num_quantizers = self.config["model_config"]['denoise_config']['num_hirearchy'],
                decay = self.config["model_config"]['denoise_config']['ema']
            ).to(self.config["device"])
        self.begin_adv = self.config["model_config"]['denoise_config']['begin_adv']
        self.user_interact_history = self.dataset.get_interaction_matrix(self.device)
        self.prev_centroids = []
    
    def save_previous_codebooks(self): 
        self.prev_centroids = []
        all_items = self.model.get_all_item_emb()
        _, _, _ = self.codebook(all_items)
        for layer in self.codebook.layers:
            self.prev_centroids.append(layer.codebook.clone().detach())
    
    def _train_epoch(self, epoch):
        start_t = time.time()
        epoch_loss = 0
        
        for batch_data in self.dataloader:
            self.model.train()
            user_id_list, pos_item_list, neg_item_list = self.dataset.get_train_batch(batch_data)
            if self.config["model"] in ["NeuMF"]:
                pos_logits, neg_logits, _, _, _, l2_norm_sq = self.model(user_id_list, pos_item_list, neg_item_list)
                pos_logits = torch.sum(pos_logits, dim=1)
                neg_logits = torch.sum(neg_logits, dim=1)
            else:
                users_emb, pos_items_emb, neg_items_emb, l2_norm_sq, all_items = self.model.forward_vq(user_id_list, pos_item_list, neg_item_list)
                pos_logits = torch.sum(users_emb * pos_items_emb, dim=1)
                neg_logits = torch.sum(users_emb * neg_items_emb, dim=1)
            
            if epoch >= self.begin_adv:
                temp_codebook = copy.deepcopy(self.codebook)
                quantized_item, quantized_idx, _ = temp_codebook(all_items)  # idx: [num_hirec, N]
                # 아이템 코드북 인덱스 (1차 레이어 기준)
                first_layer_idx = quantized_idx.T[0]  # shape: [num_items]
                
                # 아이템 → 클러스터 one-hot 매핑 (num_items, num_clusters)
                item_cluster_onehot = F.one_hot(first_layer_idx, num_classes=self.num_codebook).float()  # [N, C]
                user_ids = torch.tensor(user_id_list, device=self.device) # scipy indexing은 numpy 필요
               
                # 유저별 soft histogram 계산 (B, N) @ (N, C) = (B, C)
                user_hist_all = torch.sparse.mm(self.user_interact_history, item_cluster_onehot)
                user_hist = user_hist_all[user_ids].to(self.device)  # (B, C)

                # Normalize to get attention weights
                #user_attn = torch.softmax(user_hist, dim=1)  # (B, C)
                user_attn = user_hist / (user_hist.sum(dim=1, keepdim=True) + 1e-8)  # (B, C)

                # --- codebook 기반 user centroid ---
                codebook_vecs = temp_codebook.layers[0].codebook  # [K, D]
                user_centroid = torch.matmul(user_attn, codebook_vecs)  # (B, D)
              
                # --- 유사도 기반 soft weight 계산 ---
                user_centroid = F.normalize(user_centroid, dim=1)
                pos_vec = F.normalize(pos_items_emb, dim=1)
                cosine_sim = torch.sum(user_centroid * pos_vec, dim=1)  # (B,)

                # --- cluster 안정도: 현재 클러스터 변화량 기반 ---
            
                prev = self.prev_centroids[0]
                curr = temp_codebook.layers[0].codebook.detach()
                delta = torch.norm(curr - prev, dim=1)
                #norm_delta = delta / sum(delta)
                norm_delta = (delta - delta.min()) / (delta.max() - delta.min() + 1e-8)
                inv_delta = 1.0 - norm_delta  # 안정도

                pos_code_idx = quantized_idx.T[0][pos_item_list]
                cluster_stability = inv_delta[pos_code_idx]  # (B,)

                # 최종 weight = 안정도 × 유사도
                weights = cluster_stability * cosine_sim
                weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8).to(self.device).detach()
                    
            else:
                weights = torch.ones(len(pos_item_list), dtype=torch.float).to(self.device).detach()
            
            self.opt.zero_grad()
            loss = (self._rec_loss(pos_logits, neg_logits) * weights).mean() + self.config["weight_decay"] * l2_norm_sq
            # loss = (self._rec_loss(pos_logits, neg_logits) * weights).sum() / weights.sum()
            # loss =  loss.mean() + self.config["weight_decay"] * l2_norm_sq
            loss.backward()
            self.opt.step()
            epoch_loss += loss.item()

        if epoch >= self.begin_adv-1:
            # Update codebook
            self.save_previous_codebooks()
        
        end_t = time.time()
        print(f"Epoch {epoch}: Rec Loss: {epoch_loss/len(self.dataloader):.4f}, Time: {end_t-start_t:.2f}")

class BODCFTrainer(CFTrainer):
    def __init__(self, trainer_config):
        super().__init__(trainer_config)
        
        # BOD-specific config
        self.generator_lr = 0.001
        self.generator_reg = 0.0001
        self.weight_bpr = 1
        self.weight_alignment = 1
        self.weight_uniformity = 1
        self.outer_loop = 1
        self.inner_loop = 1

        # Generator init
        self.model_generator = GraphGenerator_VAE(self.dataset, emb_size=64).to(self.device)
        self.generator_opt = torch.optim.Adam(self.model_generator.parameters(), lr=self.generator_lr)

        # Save backbone model reference for gradient matching
        self.model_parameters = list(self.model.parameters())
    
    def bpr_loss_weight(self, user_emb, pos_item_emb, neg_item_emb, weight_pos, weight_neg):
        pos_score = weight_pos*torch.mul(user_emb, pos_item_emb).sum(dim=1)
        neg_score = weight_neg*torch.mul(user_emb, neg_item_emb).sum(dim=1)
        loss = -torch.log(10e-8 + torch.sigmoid(pos_score - neg_score))
        return torch.mean(loss)
 
    def alignment_loss_weight_1(self, x, y, weight, alpha=2):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        loss = (x - y).norm(p=2, dim=1).pow(alpha)
        return (weight*loss).mean()

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

        # TODO: output node!!!!
        if len(gwr.shape) == 2:
            gwr = gwr.T
            gws = gws.T
        if len(shape) == 4: # conv, out*in*h*w
            gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
            gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
        elif len(shape) == 3:  # layernorm, C*h*w
            gwr = gwr.reshape(shape[0], shape[1] * shape[2])
            gws = gws.reshape(shape[0], shape[1] * shape[2])
        elif len(shape) == 2: # linear, out*in
            tmp = 'do nothing'
        elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
            gwr = gwr.reshape(1, shape[0])
            gws = gws.reshape(1, shape[0])
            return 0

        dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
        dis = dis_weight
        return dis
    
    def match_loss(self, gw_syn, gw_real, dis_metric):
        dis = torch.tensor(0.0).to('cuda')
        if dis_metric == 'ours':
            for ig in range(len(gw_real)):
                gwr = gw_real[ig]
                gws = gw_syn[ig]
                dis += self.distance_wb(gwr, gws)
        else:
            exit('DC error: unknown distance function')
        return dis

    def _train_epoch(self, epoch):
        # === Inner Loop ===
        for inner_iter in range(self.inner_loop):
            for batch in self.dataloader:
                user_id_list, pos_item_list, neg_item_list = self.dataset.get_train_batch(batch)
                self.model.train()
                self.opt.zero_grad()
                self.generator_opt.zero_grad()

                # forward backbone
                user_emb, item_emb = self.model._get_rep()
                u_emb = user_emb[user_id_list]
                i_emb = item_emb[pos_item_list]
                j_emb = item_emb[neg_item_list]

                # generator weight
                w_pos = self.model_generator(u_emb, i_emb).detach()
                w_neg = self.model_generator(u_emb, j_emb).detach()

                # loss
                loss_bpr = self.bpr_loss_weight(u_emb, i_emb, j_emb, w_pos, w_neg)
                loss_align = self.alignment_loss_weight_1(u_emb, i_emb, w_pos)
                loss_uniform = (self.uniformity_loss(u_emb) + self.uniformity_loss(i_emb)) / 2

                loss = self.weight_bpr * loss_bpr + self.weight_alignment * loss_align + self.weight_uniformity * loss_uniform

                loss.backward()
                self.opt.step()

        # === Outer Loop (Generator update) ===
        for _ in range(self.outer_loop):
            self.model.eval()
            user_emb, item_emb = self.model._get_rep()
            rand_user_list = np.random.randint(0, self.dataset.n_users, size=2048)
            batch = self.dataset.get_train_batch(rand_user_list)
            u, i, j = batch
            u_emb = user_emb[u]
            i_emb = item_emb[i]
            j_emb = item_emb[j]

            w_pos = self.model_generator(u_emb, i_emb)
            w_neg = self.model_generator(u_emb, j_emb)

            loss_real = self.bpr_loss_weight(u_emb, i_emb, j_emb, w_pos.detach(), w_neg.detach())
            gw_real = torch.autograd.grad(loss_real, self.model_parameters, retain_graph=True, create_graph=True)

            # synthetic gradient via alignment
            align_syn = self.alignment_loss_weight_1(u_emb, i_emb, w_pos)
            gw_syn = torch.autograd.grad(align_syn, self.model_parameters, retain_graph=True, create_graph=True)

            loss_match = self.match_loss(gw_syn, gw_real, dis_metric="ours")
            loss_reg = self.l2_reg_loss(self.generator_reg, u_emb, i_emb)
            loss = loss_match + loss_reg

            self.generator_opt.zero_grad()
            loss.backward()
            self.generator_opt.step()

        print(f"Epoch {epoch} done.")