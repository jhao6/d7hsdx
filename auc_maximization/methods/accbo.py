from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import SGD
import gc
import torch
# from sklearn.metrics import aucuracy_score
import numpy as np
from .RNN_net import RNN
import copy
from aucloss import AUCMLoss, roc_auc_score


import math
GLOVE_DIM = 300

class Learner(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, training_size):
        """
        :param args:
        """
        super(Learner, self).__init__()
        self.args = args
        self.num_labels = args.num_labels
        self.xi = args.xi
        self.data=args.data
        self.outer_batch_size = args.outer_batch_size
        self.inner_batch_size = args.inner_batch_size
        self.outer_update_lr = args.outer_update_lr
        self.old_outer_update_lr = args.outer_update_lr
        self.inner_update_lr = args.inner_update_lr
        self.inner_update_step = args.inner_update_step
        self.inner_update_step_eval = args.inner_update_step_eval
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.collate_pad_ = self.collate_pad if args.data=='news_data' else self.collate_pad_snli
        self.training_size = training_size
        self.interval = args.interval
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
        # self.lambda_x =  torch.ones((self.training_size)).to(self.device)
        # self.lambda_x.requires_grad=True
        param_count = 0
        for param in self.inner_model.parameters():
            param_count += param.numel()
        self.a = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.b = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.alpha = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.upper_variables= [self.a, self.b]+list(self.inner_model.parameters())
        self.hyper_momentum = [torch.zeros(param.size()).to(self.device) for param in
                                       self.upper_variables]
        self.neumann_series_history = [torch.zeros(param.size()).to(self.device) for param in
                                       self.upper_variables]
        self.y = self.alpha.detach().clone().to(self.device)
        self.y_hat = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.outer_optimizer = SGD(self.upper_variables, lr=self.outer_update_lr)
        self.inner_optimizer = SGD([self.alpha], lr=self.inner_update_lr)
        self.inner_stepLR = torch.optim.lr_scheduler.StepLR(self.inner_optimizer, step_size=args.epoch, gamma=0.2)
        self.outer_stepLR = torch.optim.lr_scheduler.StepLR(self.outer_optimizer, step_size=args.epoch, gamma=0.2)
        self.aucloss = AUCMLoss(self.a, self.b, self.alpha)
        self.inner_model.train()
        self.gamma = args.gamma
        self.beta = args.beta
        self.nu = args.nu
        self.y_warm_start = args.y_warm_start
        self.normalized = args.grad_normalized
        self.grad_clip = args.grad_clip
        self.no_meta = args.no_meta
        self.criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)

    def forward(self, train_loader, val_loader, training=True, epoch=0):
        task_aucs = []
        task_loss = []
        self.inner_model.to(self.device)
        # self.y_warm_start = math.ceil(self.y_warm_start/2) if (task_id%5==0 and task_id>0) else self.y_warm_start
        num_inner_update_step = self.y_warm_start if epoch == 0  else self.inner_update_step
        warm_start = False
        for step, data in enumerate(train_loader):
            if epoch == 0:
                for i in range(0, num_inner_update_step):
                    self.inner_optimizer.zero_grad()
                    # input, label_id, data_indx = next(iter(train_loader))
                    input, label_id, data_indx = next(iter(train_loader))
                    outputs = predict(self.inner_model, input)
                    inner_loss = -self.aucloss(outputs, label_id.to(self.device))
                    inner_loss.backward()
                    self.inner_optimizer.step()
                if warm_start == False:
                    self.y_hat = self.alpha.data.clone()
                    self.y = self.y_hat.data.clone()
                    warm_start = True
            self.inner_optimizer.zero_grad()
            old_alpha = self.alpha.data.clone()
            self.alpha.data += self.gamma * (self.alpha.data - self.y.data)
            self.y = old_alpha
            input, label_id, data_indx = next(iter(train_loader))
            self.inner_optimizer.zero_grad()
            outputs = predict(self.inner_model, input)
            inner_loss = -self.aucloss(outputs, label_id.to(self.device))
            inner_loss.backward()
            self.inner_optimizer.step()
            self.inner_optimizer.zero_grad()
            self.outer_optimizer.zero_grad()
            print(f'inner loss: {inner_loss.item():.4f}')
            # self.y = self.alpha.data.clone()
            self.y_hat = (1-self.args.tau) * self.y_hat + self.args.tau * self.alpha.data
            temp_alpha = self.alpha.data.clone()
            self.alpha.data = self.y_hat.data.clone()
            input_val, label_id_val, data_indx_val = next(iter(val_loader))
            outputs = predict(self.inner_model, input_val)
            outer_loss = self.aucloss(outputs, label_id_val.to(self.device))

            self.neumann_series(outer_loss, next(iter(train_loader)), next(iter(train_loader)))
            self.alpha.data = temp_alpha
            for i, param in enumerate(self.upper_variables):
                param.grad = self.hyper_momentum[i]/self.hyper_momentum[i].norm()

            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad()

            logits = F.softmax(outputs, dim=1)[:, -1]
            label_id = label_id_val.detach().cpu().numpy().tolist()

            auc = roc_auc_score(label_id, logits.detach().cpu().numpy())
            task_aucs.append(auc)
            task_loss.append(outer_loss.detach().cpu().numpy())
            torch.cuda.empty_cache()
            # print(f'Task loss: {np.mean(task_loss):.4f}')
            print(f'Task loss: {outer_loss.detach().item():.4f}, Task auc: {auc:.4f}')
        print(f'step={step}')
        # self.inner_stepLR.step()
        # self.outer_stepLR.step()
        return np.mean(task_aucs),  np.mean(task_loss)

    def test(self, test_loader):
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


    def neumann_series(self, outer_loss, train_data_batch, val_data_batch):
            input_tr, label_id_tr, data_indx_tr = train_data_batch
            input_val, label_id_val, data_indx_val = val_data_batch
            Fy_gradient = torch.autograd.grad(outer_loss, self.alpha, retain_graph=True)
            v_0 = Fy_gradient[0].data.detach()

            z_list = []
            Fx_gradient = torch.autograd.grad(outer_loss, self.upper_variables)
            self.inner_optimizer.zero_grad()
            self.outer_optimizer.zero_grad()
            output = predict(self.inner_model, input_tr)
            inner_loss = -self.aucloss(output, label_id_tr.to(self.device))
            Gy_gradient = torch.autograd.grad(inner_loss, self.alpha, create_graph=True)
            G_gradient = []
            for g_grad, param in zip(Gy_gradient, self.alpha):
                G_gradient.append((param - self.args.hessian_lr * g_grad).view(-1))
            for _ in range(self.args.hessian_q):
                v_new = torch.autograd.grad(G_gradient, self.alpha, grad_outputs=v_0, retain_graph=True)
                v_0 = v_new[0].data.detach()
                z_list.append(v_0)
            index = np.random.randint(self.args.hessian_q)
            v_Q = self.args.hessian_lr * z_list[index]
            # v_Q = self.args.hessian_lr * v_0 + torch.sum(torch.stack(z_list), dim=0)


            output = predict(self.inner_model, input_val)
            inner_loss = -self.aucloss(output, label_id_val.to(self.device))
            Gy_gradient = torch.autograd.grad(inner_loss, self.alpha, retain_graph=True, create_graph=True)
            Gyxv_gradients = torch.autograd.grad(Gy_gradient, self.upper_variables, grad_outputs=v_Q,  allow_unused=True)
            # Gyxv_gradients = [torch.zeros(1).cuda(), torch.zeros(1).cuda()] + [i for i in Gyxv_gradient]
            for i, (f_x, g_yxv) in enumerate(zip(Fx_gradient, Gyxv_gradients)):
                if g_yxv is not None:
                    current_neumann_series = (f_x - g_yxv).data.clone()
                else:
                    current_neumann_series = f_x.data.clone()
                self.hyper_momentum[i] = self.args.beta * self.hyper_momentum[i] + (1 - self.args.beta) * current_neumann_series \
                                        + self.args.beta * (current_neumann_series - self.neumann_series_history[i] )
                self.neumann_series_history[i] = current_neumann_series

    def hypergradient(self, args, jacob_flat, loss, query_batch):
        val_data, val_labels, data_idx = query_batch
        loss.backward()

        # Fy_gradient = torch.autograd.grad(loss, adapter_model.parameters(), retain_graph=True)
        Fy_gradient = [g_param.grad.detach().view(-1) for g_param in self.inner_model.parameters()]
        Fy_gradient_flat = torch.unsqueeze(torch.reshape(torch.hstack(Fy_gradient), [-1]), 1)
        self.z_params -= args.nu * (jacob_flat - Fy_gradient_flat)
        # Fx_gradient = torch.autograd.grad(loss, self.lambda_x)

        # Gyx_gradient
        output = predict(self.inner_model, val_data)
        loss = torch.mean(
            torch.sigmoid(self.lambda_x[data_idx]) * F.cross_entropy(output, val_labels.cuda(), reduction='none')) + 0.0001 * sum(
            [x.norm().pow(2) for x in self.inner_model.parameters()]).sqrt()
        Gy_gradient = torch.autograd.grad(loss, self.inner_model.parameters(), retain_graph=True, create_graph=True)
        Gy_params = [Gy_param.view(-1) for Gy_param in Gy_gradient]
        Gy_gradient_flat = torch.reshape(torch.hstack(Gy_params), [-1])
        Gyxz_gradient = torch.autograd.grad(-torch.matmul(Gy_gradient_flat, self.z_params.detach()), self.lambda_x)
        self.hyper_momentum = [args.beta * h + (1 - args.beta) *  Gyxz for (h, Gyxz) in
                          zip(self.hyper_momentum,  Gyxz_gradient)]

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



