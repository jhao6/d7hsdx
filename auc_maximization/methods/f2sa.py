from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import SGD
import gc
import torch
from sklearn.metrics import accuracy_score
import numpy as np
from .RNN_net import RNN
from aucloss import AUCMLoss, roc_auc_score
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
        self.outer_batch_size = args.outer_batch_size
        self.inner_batch_size = args.inner_batch_size
        self.outer_update_lr = args.outer_update_lr
        self.old_outer_update_lr = args.outer_update_lr
        self.inner_update_lr = args.inner_update_lr
        self.inner_update_step = args.inner_update_step
        self.inner_update_step_eval = args.inner_update_step_eval
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lamb = args.lamb
        self.nu = args.nu
        self.training_size = training_size
        if args.data == 'news_data':
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

        self.a = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.b = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.alpha = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.z = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.upper_variables = [self.a, self.b] + list(self.inner_model.parameters())
        self.outer_optimizer = SGD(self.upper_variables, lr=self.outer_update_lr)
        self.inner_optimizer_y = SGD([self.alpha], lr=self.inner_update_lr)
        self.inner_optimizer_z = SGD([self.z], lr=self.inner_update_lr)
        self.inner_stepLR_y = torch.optim.lr_scheduler.StepLR(self.inner_optimizer_y, step_size=args.epoch, gamma=0.2)
        self.inner_stepLR_z = torch.optim.lr_scheduler.StepLR(self.inner_optimizer_z, step_size=args.epoch, gamma=0.2)
        self.outer_stepLR = torch.optim.lr_scheduler.StepLR(self.outer_optimizer, step_size=args.epoch, gamma=0.2)
        self.aucloss_y = AUCMLoss(self.a, self.b, self.alpha)
        self.aucloss_z = AUCMLoss(self.a, self.b, self.z)
        self.inner_model.train()
        self.gamma = args.gamma
        self.beta = args.beta
        self.nu = args.nu
        self.normalized = args.grad_normalized
        self.no_meta = args.no_meta
        self.criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)

    def forward(self, train_loader, val_loader, training=True, epoch=0):
        task_aucs = []
        task_loss = []
        sum_gradients = []
        num_inner_update_step = self.inner_update_step

        for step, data in enumerate(train_loader):
            self.inner_model.to(self.device)

            all_loss = []

            input_y, label_id_y, data_indx_y = data
            input_z, label_id_z, data_indx_z = next(iter(train_loader))
            input_, label_id_, data_indx_ = next(iter(val_loader))
            outputs_y = predict(self.inner_model, input_y)
            outputs_z = predict(self.inner_model, input_z)
            outputs_ = predict(self.inner_model, input_)
            inner_loss_z =  -self.aucloss_z(outputs_z, label_id_z.to(self.device))
            inner_loss_y =  -self.aucloss_y(outputs_y, label_id_y.to(self.device))
            outer_loss_  =  self.aucloss_y(outputs_, label_id_.to(self.device))

            inner_loss_z.backward()
            self.inner_optimizer_z.step()
            self.inner_optimizer_z.zero_grad()
            self.inner_optimizer_y.zero_grad()
            gy_grad = torch.autograd.grad(inner_loss_y, self.alpha)
            fy_grad = torch.autograd.grad(outer_loss_,  self.alpha)
            self.alpha.grad = fy_grad[0].detach() + self.lamb * gy_grad[0].detach()
            self.inner_optimizer_y.step()
            self.inner_optimizer_y.zero_grad()

            self.outer_optimizer.zero_grad()
            input, label_id, data_indx = next(iter(train_loader))
            outputs = predict(self.inner_model, input)
            loss = -self.aucloss_y(outputs, label_id.to(self.device))
            g_xy = torch.autograd.grad(loss, self.upper_variables)

            input, label_id, data_indx = next(iter(train_loader))
            outputs = predict(self.inner_model, input)
            loss = -self.aucloss_z(outputs, label_id.to(self.device))
            g_xz = torch.autograd.grad(loss, self.upper_variables)

            q_input, q_label_id, q_data_indx = data
            q_outputs = predict(self.inner_model, q_input)
            outer_loss = self.aucloss_y(q_outputs, q_label_id.to(self.device))
            fx = torch.autograd.grad(outer_loss, self.upper_variables)


            for i, param in enumerate(self.upper_variables):
                param.grad = fx[i].detach() + self.lamb * (g_xy[i].detach()- g_xz[i].detach())
            # self.alpha.grad =  self.lamb * (g_xy[0]- g_xz[0])
            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad()


            q_logits = F.softmax(q_outputs, dim=1)[:, -1]
            q_label_id = q_label_id.detach().cpu().numpy().tolist()
            auc = roc_auc_score(q_label_id, q_logits.detach().cpu().numpy())
            task_aucs.append(auc)
            task_loss.append(outer_loss.detach().cpu().numpy())
            self.outer_optimizer.zero_grad()
            torch.cuda.empty_cache()
            print(f'Task loss: {outer_loss.detach().item():.4f}, Task auc: {auc:.4f}')

        self.lamb += 1
        self.inner_stepLR_y.step()
        self.inner_stepLR_z.step()
        self.outer_stepLR.step()
        return np.mean(task_aucs), np.mean(task_loss)

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
            task_loss.append(q_loss.detach().cpu())
            torch.cuda.empty_cache()
            print(f'Task loss: {np.mean(task_loss):.4f}, Task acc: {np.mean(task_aucs):.4f}')
        return np.mean(task_aucs), np.mean(task_loss)

    def collate_pad_(self, data_points):
        """ Pad data points with zeros to fit length of longest data point in batch. """
        s_embeds = data_points[0] if type(data_points[0]) == list else data_points[1]
        targets = data_points[1] if type(data_points[0]) == list else data_points[0]

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

def predict(net, inputs):
    """ Get predictions for a single batch. """
    s_embed, s_lens = inputs
    outputs = net((s_embed.cuda(), s_lens))
    return outputs



