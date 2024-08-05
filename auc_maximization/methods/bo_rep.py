from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import SGD
import gc
import torch
from sklearn.metrics import accuracy_score
import numpy as np
from .RNN_net import RNN
import copy
import math
from aucloss import AUCMLoss, roc_auc_score
GLOVE_DIM = 300

class Learner(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args):
        """
        :param args:
        """
        super(Learner, self).__init__()
        self.args = args
        self.outer_update_lr = args.outer_update_lr
        self.inner_update_lr = args.inner_update_lr
        self.inner_update_step = args.inner_update_step
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.collate_pad_ = self.collate_pad if args.data=='news_data' else self.collate_pad_snli
        self.update_interval = args.update_interval
        if args.data == 'news_data':
            self.meta_model = RNN(
                word_embed_dim=args.word_embed_dim,
                encoder_dim=args.encoder_dim,
                n_enc_layers=args.n_enc_layers,
                dpout_model=0.0,
                dpout_fc=0.0,
                fc_dim=args.fc_dim,
                n_classes=args.n_classes,
                pool_type=args.pool_type,
                linear_fc=args.linear_fc
            )
        if args.data == 'snli' or 'sentment140':
            self.inner_model = RNN(
                word_embed_dim=args.word_embed_dim,
                encoder_dim=args.encoder_dim,
                n_enc_layers=args.n_enc_layers,
                dpout_model=0.0,
                dpout_fc=0.0,
                fc_dim=args.fc_dim,
                n_classes=args.n_classes,
                pool_type=args.pool_type,
                linear_fc=args.linear_fc
            )
        param_count = 0
        for param in self.inner_model.parameters():
            param_count += param.numel()
        self.a = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.b = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.alpha = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.upper_variables= [self.a, self.b]+list(self.inner_model.parameters())
        self.hyper_momentum = [torch.zeros(param.size()).to(self.device) for param in
                                       self.upper_variables]
        self.z_params = torch.randn(1, 1)
        self.z_params = nn.init.xavier_uniform_(self.z_params).to(self.device)[0]

        self.outer_optimizer = SGD(self.upper_variables, lr=self.outer_update_lr)
        self.inner_optimizer = SGD([self.alpha], lr=self.inner_update_lr)
        self.inner_stepLR = torch.optim.lr_scheduler.StepLR(self.inner_optimizer, step_size=args.epoch, gamma=0.2)
        self.outer_stepLR = torch.optim.lr_scheduler.StepLR(self.outer_optimizer, step_size=args.epoch, gamma=0.2)
        self.aucloss = AUCMLoss(self.a, self.b, self.alpha)
        self.inner_model.train()
        self.beta = args.beta
        self.nu = args.nu
        self.y_warm_start = args.y_warm_start
        self.criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)

    def forward(self, train_loader, val_loader, training=True, epoch=0):
        task_accs = []
        task_loss = []
        self.inner_model.to(self.device)
        # self.y_warm_start = math.ceil(self.y_warm_start/2) if (task_id%5==0 and task_id>0) else self.y_warm_start
        for step, data in enumerate(train_loader):
            num_inner_update_step = self.y_warm_start if step%self.update_interval==0  else self.inner_update_step
            all_loss = []

            input, label_id, data_indx = data
            outputs = predict(self.inner_model, input)
            inner_loss =  -self.aucloss(outputs, label_id.to(self.device))
            inner_loss.backward(retain_graph=True, create_graph=True)
            all_loss.append(inner_loss.item())
            g_grad = self.alpha.grad
            jacob = torch.autograd.grad(g_grad, self.alpha, grad_outputs=self.z_params)
            if step%self.update_interval == 0:
                self.inner_optimizer.step()
                for i in range(0, num_inner_update_step):
                    input, label_id, data_indx = next(iter(train_loader))
                    outputs = predict(self.inner_model, input)
                    inner_loss =  -self.aucloss(outputs, label_id.to(self.device))
                    inner_loss.backward()
                    self.inner_optimizer.step()
                    # print(f'inner loss: {inner_loss}')
                # self.outer_optimizer.zero_grad()
            self.inner_optimizer.zero_grad()
            q_input, q_label_id, q_data_indx = next(iter(val_loader))
            q_outputs = predict(self.inner_model, q_input)
            q_loss = self.aucloss(q_outputs, q_label_id.to(self.device))
            train_batch = next(iter(train_loader))
            self.hypergradient(self.args, jacob[0], q_loss, train_batch)
            for i, param in enumerate(self.upper_variables):
                param.grad = self.hyper_momentum[i]/self.hyper_momentum[i].norm()

            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad()
            # if self.normalized:
            # self.outer_optimizer.param_groups[0][ 'lr'] = old_out_lr
            q_logits = F.softmax(q_outputs, dim=1)[:, -1]
            q_label_id = q_label_id.detach().cpu().numpy().tolist()
            self.outer_optimizer.zero_grad()
            auc = roc_auc_score(q_label_id,  q_logits.detach().cpu().numpy())
            task_accs.append(auc)
            task_loss.append(q_loss.detach().cpu().numpy())
            torch.cuda.empty_cache()
            print(f'Task loss: {q_loss.detach().cpu().item():.4f}, Task auc: {auc:.4f}')
        # self.inner_stepLR.step()
        # self.outer_stepLR.step()
        return np.mean(task_accs),  np.mean(task_loss)

    def test(self, test_loader):
        task_accs = []
        task_loss = []

        self.inner_model.to(self.device)
        for step, data in enumerate(test_loader):
            task_aucs = []
            task_loss = []

            self.inner_model.to(self.device)
            for step, data in enumerate(test_loader):
                q_input, q_label_id, q_data_indx = data
                q_outputs = predict(self.inner_model, q_input)
                q_loss = self.aucloss(q_outputs, q_label_id.to(self.device))

                q_logits = F.softmax(q_outputs, dim=1)[:, -1]
                q_label_id = q_label_id.detach().cpu().numpy().tolist()
                auc = roc_auc_score(q_label_id, q_logits.detach().cpu().numpy())
                task_aucs.append(auc)
                task_loss.append(q_loss.detach().cpu().numpy())
                torch.cuda.empty_cache()
                print(f'Task loss: {q_loss.detach().cpu().item():.4f}, Task auc: {auc:.4f}')
            return np.mean(task_aucs), np.mean(task_loss)

    def hypergradient(self, args, jacob_flat, loss, data_batch):
        data, labels, data_idx = data_batch
        Fy_gradient = torch.autograd.grad(loss, self.alpha, retain_graph=True)
        Fx_gradient = torch.autograd.grad(loss, self.upper_variables)
        self.z_params -= args.nu * (jacob_flat.detach() - Fy_gradient[0].data)

        # Gyx_gradient
        output = predict(self.inner_model, data)
        loss = -self.aucloss(output, labels.to(self.device))
        Gy_gradient = torch.autograd.grad(loss, self.alpha, retain_graph=True, create_graph=True)
        Gyxz_gradient = torch.autograd.grad(Gy_gradient, self.upper_variables, grad_outputs=self.z_params.detach(), allow_unused=True)
        self.hyper_momentum = [args.beta * h + (1 - args.beta) *  (fx.detach()-Gyxz.detach() if Gyxz is not None else fx.detach()) for (h, fx, Gyxz) in
                          zip(self.hyper_momentum, Fx_gradient,  Gyxz_gradient)]

        # def test():

    def collate_pad(self, data_points):
        """ Pad data points with zeros to fit length of longest data point in batch. """
        s_embeds = data_points[0] if type(data_points[0]) == list or type(data_points[0]) == tuple else data_points[1]
        targets = data_points[1] if type(data_points[0]) == list or type(data_points[0]) == tuple  else data_points[0]

        # Get sentences for batch and their lengths.
        s_lens = np.array([sent.shape[0] for sent in s_embeds])
        max_s_len = np.max(s_lens)
        # Encode sentences as glove vectors.
        bs = len(data_points[0])
        s_embed = np.zeros((max_s_len, bs, GLOVE_DIM))
        for i in range(bs):
            e = s_embeds[i]
            if len(e) <= 0:
                s_lens[i] = 1
            s_embed[: len(e), i] = e.copy()
        embeds = torch.from_numpy(s_embed).float().to(self.device)
        targets = torch.LongTensor(targets).to(self.device)
        return (embeds, s_lens), targets

    def collate_pad_snli(self, data_points):
        """ Pad data points with zeros to fit length of longest data point in batch. """
        s_embeds = data_points[0] if type(data_points[0]) == list or type(data_points[0]) == tuple else data_points[1]
        # s_embeds = data_points[0]
        # s2_embeds = data_points[0] if type(data_points[0]) == list or type(data_points[0]) == tuple else data_points[1]
        targets = data_points[1] if type(data_points[0]) == list or type(data_points[0]) == tuple else data_points[0]
        # targets = data_points[1]
        s1_embeds = [x for x in s_embeds[0]]
        s2_embeds = [x for x in s_embeds[1]]
        # targets = [x[1] for x in data_points]

        # Get sentences for batch and their lengths.
        s1_lens = np.array([sent.shape[0] for sent in s1_embeds])
        max_s1_len = np.max(s1_lens)
        s2_lens = np.array([sent.shape[0] for sent in s2_embeds])
        max_s2_len = np.max(s2_lens)
        lens = (s1_lens, s2_lens)

        # Encode sentences as glove vectors.
        bs = len(targets)
        s1_embed = np.zeros((max_s1_len, bs, GLOVE_DIM))
        s2_embed = np.zeros((max_s2_len, bs, GLOVE_DIM))
        for i in range(bs):
            e1 = s1_embeds[i]
            e2 = s2_embeds[i]
            s1_embed[: len(e1), i] = e1.copy()
            s2_embed[: len(e2), i] = e2.copy()
            if len(e1) <= 0:
                s1_lens[i] = 1
            if len(e2) <= 0:
                s2_lens[i] = 1
        embeds = (
            torch.from_numpy(s1_embed).float().to(self.device), torch.from_numpy(s2_embed).float().to(self.device)
        )

        # Convert targets to tensor.
        targets = torch.LongTensor(targets).to(self.device)

        return (embeds, lens), targets

def predict(net, inputs):
    """ Get predictions for a single batch. """
    # snli dataaset
    # (s1_embed, s2_embed), (s1_lens, s2_lens) = inputs
    # outputs = net((s1_embed.cuda(), s1_lens), (s2_embed.cuda(), s2_lens))
    s_embed, s_lens = inputs
    outputs = net((s_embed.cuda(), s_lens))
    return outputs



