import sys
import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from sklearn.metrics import roc_auc_score
from layers import Dense, CNN_cross,GCN,Dense2,HCN


class MKR_model(nn.Module):
    def __init__(self, args, n_user, n_item, n_entity, n_relation, use_inner_product=True):
        super(MKR_model, self).__init__()

        # <Lower Model>
        self.args = args
        self.n_user = n_user
        self.n_item = n_item
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.use_inner_product = use_inner_product
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Init embeddings
        self.user_embeddings_lookup = nn.Embedding(self.n_user, self.args.dim)
        self.item_embeddings_lookup = nn.Embedding(self.n_item, self.args.dim)
        self.entity_embeddings_lookup = nn.Embedding(self.n_entity, self.args.dim)
        self.relation_embeddings_lookup = nn.Embedding(self.n_relation, self.args.dim)

        self.user_mlp = nn.Sequential()
        self.G_mlp = nn.Sequential()
        self.G_mlp2 = nn.Sequential()
        self.G_mlp3 = nn.Sequential()
        self.G_mlp4 = nn.Sequential()
        self.G2_mlp = nn.Sequential()
        self.tail_mlp = nn.Sequential()
        self.cc_unit = nn.Sequential()
        self.gcn_unit = nn.Sequential()
        self.hcn_unit = nn.Sequential()
        for i_cnt in range(self.args.L):
            self.user_mlp.add_module('user_mlp{}'.format(i_cnt),
                                     Dense(self.args.dim, self.args.dim))
            self.G_mlp.add_module('G_mlp{}'.format(i_cnt),
                                     Dense2(self.args.dim, self.args.dim))
            self.G_mlp2.add_module('G_mlp2{}'.format(i_cnt),
                                  Dense2(self.args.dim, self.args.dim))
            self.G_mlp3.add_module('G_mlp3{}'.format(i_cnt),
                                  Dense2(self.args.dim, self.args.dim))
            self.G_mlp4.add_module('G_mlp4{}'.format(i_cnt),
                                  Dense2(self.args.dim, self.args.dim))
            self.G2_mlp.add_module('G2_mlp{}'.format(i_cnt),
                                  Dense2(self.args.dim*2, self.args.dim))
            self.tail_mlp.add_module('tail_mlp{}'.format(i_cnt),
                                     Dense(self.args.dim, self.args.dim))
            self.cc_unit.add_module('cc_unit{}'.format(i_cnt),
                                     CNN_cross(self.args.dim))
            self.gcn_unit.add_module('gcn_unit{}'.format(i_cnt),
                                    GCN(self.args.dim,self.args.batch_size))
            self.hcn_unit.add_module('hcn_unit{}'.format(i_cnt),
                                     HCN(self.args.dim, self.args.batch_size))
        # <Higher Model>
        self.kge_pred_mlp = Dense(self.args.dim * 2, self.args.dim)
        self.kge_mlp = nn.Sequential()
        for i_cnt in range(self.args.H - 1):
            self.kge_mlp.add_module('kge_mlp{}'.format(i_cnt),
                                    Dense(self.args.dim * 2, self.args.dim * 2))
        if self.use_inner_product==False:
            self.rs_pred_mlp = Dense(self.args.dim * 2, 1)
            self.rs_mlp = nn.Sequential()
            for i_cnt in range(self.args.H - 1):
                self.rs_mlp.add_module('rs_mlp{}'.format(i_cnt),
                                       Dense(self.args.dim * 2, self.args.dim * 2))

    def forward(self, user_indices=None, item_indices=None, head_indices=None,
            relation_indices=None, tail_indices=None):

        # <Lower Model>

        if user_indices is not None:
            self.user_indices = user_indices
        if item_indices is not None:
            self.item_indices = item_indices
        if head_indices is not None:
            self.head_indices = head_indices
        if relation_indices is not None:
            self.relation_indices = relation_indices
        if tail_indices is not None:
            self.tail_indices = tail_indices

        # Embeddings
        self.item_embeddings1 = self.item_embeddings_lookup(self.item_indices)

        self.head_embeddings = self.entity_embeddings_lookup(self.head_indices)

        # print('item_embeddings3',self.item_embeddings3.shape)
        # print('item_embeddings', self.item_embeddings.shape)
        # <Higher Model>
        if user_indices is not None:

            # RS
            self.item_embeddings2, self.head_embeddings = self.cc_unit([self.item_embeddings1, self.head_embeddings])
            self.item_embeddings3= self.gcn_unit([self.user_indices, self.item_indices])
            self.user_embeddings = self.user_embeddings_lookup(self.user_indices)
            self.user_embeddings = self.user_mlp(self.user_embeddings)
            self.item_embeddings_1=self.item_embeddings2+self.item_embeddings3
            self.item_embeddings_2 = self.item_embeddings2*self.item_embeddings3
            self.item_embeddings_1=self.G_mlp(self.item_embeddings_1)
            self.item_embeddings_2 = self.G_mlp2(self.item_embeddings_2)
            self.item_embeddings=self.item_embeddings_1+self.item_embeddings_2


            # self.item_embeddings=torch.cat([self.item_embeddings2,self.item_embeddings3],1)
            # self.item_embeddings=self.G2_mlp(self.item_embeddings)
            if self.use_inner_product:
                # [batch_size]
                self.scores = torch.sum(self.user_embeddings * self.item_embeddings, 1)
            else:
                # [batch_size, dim * 2]
                self.user_item_concat = torch.cat([self.user_embeddings, self.item_embeddings], 1)
                self.user_item_concat = self.rs_mlp(self.user_item_concat)
                # [batch_size]
                self.scores = torch.squeeze(self.rs_pred_mlp(self.user_item_concat))
            self.scores_normalized = torch.sigmoid(self.scores)
            outputs = [self.user_embeddings, self.item_embeddings, self.scores, self.scores_normalized]
        if relation_indices is not None:
            # KGE
            self.item_embeddings, self.head_embeddings1 = self.cc_unit([self.item_embeddings1, self.head_embeddings])
            # print('item_embeddings:KG', self.item_embeddings.shape)
            self.head_embeddings2=self.hcn_unit([self.head_indices])
            self.head_embeddings_1=self.head_embeddings1+self.head_embeddings2
            self.head_embeddings_2 = self.head_embeddings1 * self.head_embeddings2
            self.head_embeddings_1 = self.G_mlp3(self.head_embeddings_1)
            self.head_embeddings_2 = self.G_mlp4(self.head_embeddings_2)
            self.head_embeddings=self.head_embeddings_1+self.head_embeddings_2

            # self.head_embeddings = torch.cat([self.head_embeddings1, self.head_embeddings2], 1)
            # self.head_embeddings = self.G2_mlp(self.head_embeddings)

            self.tail_embeddings = self.entity_embeddings_lookup(self.tail_indices)
            self.relation_embeddings = self.relation_embeddings_lookup(self.relation_indices)
            self.tail_embeddings = self.tail_mlp(self.tail_embeddings)
            # [batch_size, dim * 2]
            self.head_relation_concat = torch.cat([self.head_embeddings, self.relation_embeddings], 1)
            self.head_relation_concat = self.kge_mlp(self.head_relation_concat)
            # [batch_size, 1]
            self.tail_pred = self.kge_pred_mlp(self.head_relation_concat)
            self.tail_pred = torch.sigmoid(self.tail_pred)
            self.scores_kge = torch.sigmoid(torch.sum(self.tail_embeddings * self.tail_pred, 1))
            self.rmse = torch.mean(
                torch.sqrt(torch.sum(torch.pow(self.tail_embeddings -
                           self.tail_pred, 2), 1) / self.args.dim))
            outputs = [self.head_embeddings, self.tail_embeddings, self.scores_kge, self.rmse]

        return outputs


class MKR(object):
    def __init__(self, args, n_users, n_items, n_entities,
                 n_relations):
        self.args = args
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._parse_args(n_users, n_items, n_entities, n_relations)
        self._build_model()
        self._build_loss()
        self._build_ops()

    def _parse_args(self, n_users, n_items, n_entities, n_relations):
        self.n_user = n_users
        self.n_item = n_items
        self.n_entity = n_entities
        self.n_relation = n_relations

    def _build_model(self):
        print("Build models")
        self.MKR_model = MKR_model(self.args, self.n_user, self.n_item, self.n_entity, self.n_relation)
        # self.MKR_model = self.MKR_model.to(self.device, non_blocking=True)
        for m in self.MKR_model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding):
                torch.nn.init.xavier_uniform_(m.weight)
        # for param in self.MKR_model.parameters():
        #     param.requires_grad = True

    def _build_loss(self):
        self.sigmoid_BCE = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

    def _build_ops(self):
        self.optimizer_rs = torch.optim.Adam(self.MKR_model.parameters(),
                                             lr=self.args.lr_rs)
        self.optimizer_kge = torch.optim.Adam(self.MKR_model.parameters(),
                                              lr=self.args.lr_kge)

    def _inference_rs(self, inputs):
        # Inputs
        self.user_indices = inputs[:, 0].long()
        self.item_indices = inputs[:, 1].long()
        labels = inputs[:, 2].float()
        self.head_indices = inputs[:, 1].long()

        # Inference
        outputs = self.MKR_model(user_indices=self.user_indices,
                                 item_indices=self.item_indices,
                                 head_indices=self.head_indices,
                                 relation_indices=None,
                                 tail_indices=None)

        user_embeddings, item_embeddings, scores, scores_normalized = outputs
        return user_embeddings, item_embeddings, scores, scores_normalized, labels

    def _inference_kge(self, inputs):
        # Inputs
        self.item_indices = inputs[:, 0].long()
        self.head_indices = inputs[:, 0].long()
        self.relation_indices = inputs[:, 1].long()
        self.tail_indices = inputs[:, 2].long()

        # Inference
        outputs = self.MKR_model(user_indices=None,
                                 item_indices=self.item_indices,
                                 head_indices=self.head_indices,
                                 relation_indices=self.relation_indices,
                                 tail_indices=self.tail_indices)

        head_embeddings, tail_embeddings, scores_kge, rmse = outputs
        return head_embeddings, tail_embeddings, scores_kge, rmse

    def l2_loss(self, inputs):
        return torch.sum(inputs ** 2) / 2

    def loss_rs(self, user_embeddings, item_embeddings, scores, labels):
        # scores_for_signll = torch.cat([1-self.sigmoid(scores).unsqueeze(1),
        #                                self.sigmoid(scores).unsqueeze(1)], 1)
        # base_loss_rs = torch.mean(self.nll_loss(scores_for_signll, labels))
        base_loss_rs = torch.mean(self.sigmoid_BCE(scores, labels))
        l2_loss_rs = self.l2_loss(user_embeddings) + self.l2_loss(item_embeddings)
        for name, param in self.MKR_model.named_parameters():
            if param.requires_grad and ('embeddings_lookup' not in name) \
                    and (('rs' in name) or ('cc_unit' in name) or ('user' in name)) \
                    and ('weight' in name):
                l2_loss_rs = l2_loss_rs + self.l2_loss(param)
        loss_rs = base_loss_rs + l2_loss_rs * self.args.l2_weight
        return loss_rs, base_loss_rs, l2_loss_rs

    def loss_kge(self, scores_kge, head_embeddings, tail_embeddings):
        base_loss_kge = -scores_kge
        l2_loss_kge = self.l2_loss(head_embeddings) + self.l2_loss(tail_embeddings)
        for name, param in self.MKR_model.named_parameters():
            if param.requires_grad and ('embeddings_lookup' not in name) \
                    and (('kge' in name) or ('tail' in name) or ('cc_unit' in name)) \
                    and ('weight' in name):
                l2_loss_kge = l2_loss_kge + self.l2_loss(param)
        # Note: L2 regularization will be done by weight_decay of pytorch optimizer
        loss_kge = base_loss_kge + l2_loss_kge * self.args.l2_weight
        return loss_kge, base_loss_kge, l2_loss_kge

    def train_rs(self, inputs, show_grad=False, glob_step=None):
        self.MKR_model.train()
        user_embeddings, item_embeddings, scores, _, labels= self._inference_rs(inputs)
        loss_rs, base_loss_rs, l2_loss_rs = self.loss_rs(user_embeddings, item_embeddings, scores, labels)

        self.optimizer_rs.zero_grad()
        loss_rs.backward()


        self.optimizer_rs.step()
        loss_rs.detach()
        user_embeddings.detach()
        item_embeddings.detach()
        scores.detach()
        labels.detach()

        return loss_rs, base_loss_rs, l2_loss_rs

    def train_kge(self, inputs, show_grad=False, glob_step=None):
        self.MKR_model.train()
        head_embeddings, tail_embeddings, scores_kge, rmse = self._inference_kge(inputs)
        loss_kge, base_loss_kge, l2_loss_kge = self.loss_kge(scores_kge, head_embeddings, tail_embeddings)

        self.optimizer_kge.zero_grad()
        loss_kge.sum().backward()

        self.optimizer_kge.step()
        loss_kge.detach()
        head_embeddings.detach()
        tail_embeddings.detach()
        scores_kge.detach()
        rmse.detach()
        return rmse, loss_kge.sum(), base_loss_kge.sum(), l2_loss_kge

    def eval(self, inputs):
        self.MKR_model.eval()
        inputs = torch.from_numpy(inputs)
        user_embeddings, item_embeddings, _, scores, labels = self._inference_rs(inputs)
        labels = labels.detach().numpy()
        scores = scores.detach().numpy()
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        acc = np.mean(np.equal(predictions, labels))

        return auc, acc
    def _get_scores(self, user, item_list, head_list):
        self.MKR_model.eval()

        # Convert numpy arrays to torch tensors
        user = torch.from_numpy(user).long()
        item_list = torch.from_numpy(item_list)
        head_list = torch.from_numpy(head_list)

        # Create a dictionary with input tensors
        inputs = {
            "user_indices": user,
            "item_indices": item_list,
            "head_indices": head_list,
            "relation_indices": None,  # Assuming relation_indices and tail_indices are not used in this context
            "tail_indices": None
        }

        with torch.no_grad():
            # Forward pass through the model
            scores_normalized = self.MKR_model(**inputs)[3]

        # Move scores to CPU and convert to NumPy array
        scores = scores_normalized.numpy()

        return scores

    def topk_eval2(self, user_list, train_record, test_record, item_set, k_list):
        print("评估 Top-K")
        precision_list = {k: [] for k in k_list}
        recall_list = {k: [] for k in k_list}
        ndcg_list = {k: [] for k in k_list}

        for user in tqdm(user_list):
            test_item_list = list(item_set - train_record[user])
            item_score_map = dict()

            scores = self._get_scores(np.array([user] * len(test_item_list)),
                                      np.array(test_item_list),
                                      np.array(test_item_list))

            items = np.array(test_item_list)
            for item, score in zip(items, scores):
                item_score_map[item] = score
            item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
            item_sorted = [i[0] for i in item_score_pair_sorted]

            for k in k_list:
                # 计算 Precision
                hit_num = len(set(item_sorted[:k]) & test_record[user])
                precision_list[k].append(hit_num / k)

                # 计算 Recall
                recall_list[k].append(hit_num / len(test_record[user]))

                # 计算 NDCG
                dcg = 0
                for i in range(k):
                    item = item_sorted[i]
                    if item in test_record[user]:
                        dcg += 1 / np.log2(i + 2)  # i+2 是为了从 1 开始计算
                idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(test_record[user]))))
                ndcg_list[k].append(dcg / idcg)

        # 计算整体指标
        precision = [np.mean(precision_list[k]) for k in k_list]
        recall = [np.mean(recall_list[k]) for k in k_list]
        ndcg = [np.mean(ndcg_list[k]) for k in k_list]
        f1 = [2 / (1 / precision[i] + 1 / recall[i]) for i in range(len(k_list))]

        return precision, recall, ndcg, f1