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
        self.y_warm_start = args.y_warm_start
        self.inner_update_lr = args.inner_update_lr
        self.inner_update_step = args.inner_update_step
        self.inner_update_step_eval = args.inner_update_step_eval
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_size = training_size
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
        self.upper_variables = [self.a, self.b] + list(self.inner_model.parameters())
        self.outer_optimizer = SGD(self.upper_variables, lr=self.outer_update_lr)
        self.inner_optimizer = SGD([self.alpha], lr=self.inner_update_lr)
        self.inner_stepLR = torch.optim.lr_scheduler.StepLR(self.inner_optimizer, step_size=args.epoch, gamma=0.2)
        self.outer_stepLR = torch.optim.lr_scheduler.StepLR(self.outer_optimizer, step_size=args.epoch, gamma=0.2)
        self.aucloss = AUCMLoss(self.a, self.b, self.alpha)
        self.inner_model.train()
        self.gamma = args.gamma
        self.grad_clip = args.grad_clip
        self.no_meta = args.no_meta
        self.criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)

    def forward(self, train_loader, val_loader, training=True, epoch=0):
        # self.model.load_state_dict(torch.load('checkpoints/itd-model.pkl'))
        task_aucs = []
        task_loss = []
        sum_gradients = []
        num_inner_update_step = self.inner_update_step

        for step, data in enumerate(train_loader):
            self.inner_model.to(self.device)
            all_loss = []

            input, label_id, data_indx = data
            for i in range(self.args.inner_update_step):
                outputs = predict(self.inner_model, input)
                loss = -self.aucloss(outputs, label_id.to(self.device))
                loss.backward()
                self.inner_optimizer.step()
            all_loss.append(loss.item())

            q_input, q_label_id, q_indx = next(iter(val_loader))
            q_outputs = predict(self.inner_model, q_input)
            q_loss = self.aucloss(q_outputs, q_label_id.to(self.device))

            hypergradient = self.stocbio(q_loss, next(iter(train_loader)), next(iter(train_loader)))
            # print(f'Task loss: {np.mean(all_loss):.4f}')
            for i, param in enumerate(self.upper_variables):
                param.grad = hypergradient[i]
            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad()

            self.inner_optimizer.zero_grad()
            logits = F.softmax(q_outputs, dim=1)[:, -1]
            label_id = q_label_id.detach().cpu().numpy().tolist()

            auc = roc_auc_score(label_id, logits.detach().cpu().numpy())
            task_aucs.append(auc)
            task_loss.append(q_loss.detach().cpu().numpy())
            torch.cuda.empty_cache()
            print(f'Task loss: {q_loss.detach().item():.4f}, Task auc: {auc:.4f}')

        # self.inner_stepLR.step()
        # self.outer_stepLR.step()

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

    def test(self, test_loader):
        task_aucs = []
        task_loss = []

        self.inner_model.to(self.device)
        for step, data in enumerate(test_loader):
            q_input, q_label_id, q_data_indx = data
            q_outputs = predict(self.inner_model, q_input)
            q_loss = self.aucloss(q_outputs, q_label_id.to(self.device))

            q_logits = F.softmax(q_outputs, dim=1)[:,-1]
            q_label_id = q_label_id.detach().cpu().numpy().tolist()
            auc = roc_auc_score(q_label_id, q_logits.detach().cpu().numpy())
            task_aucs.append(auc)
            task_loss.append(q_loss.detach().cpu().numpy())
            torch.cuda.empty_cache()
            print(f'Task loss: {np.mean(task_loss):.4f}, Task acc: {np.mean(task_aucs):.4f}')
        return np.mean(task_aucs), np.mean(task_loss)

    def stocbio(self, loss, train_data_batch, val_data_batch):
        train_data, train_labels, train_indx = train_data_batch
        val_data, val_labels, val_indx = val_data_batch
        Fy_gradient = torch.autograd.grad(loss, self.alpha, retain_graph=True)
        F_gradient = Fy_gradient[0]
        v_0 = F_gradient.detach()
        # Fx_gradient = [g_param.view(-1) for g_param in Fx_gradient]
        Fx_gradient = torch.autograd.grad(loss, self.upper_variables)
        # Hessian
        z_list = []
        outputs = predict(self.inner_model, train_data)
        inner_loss = -self.aucloss(outputs, train_labels.to(self.device))
        G_gradient = []
        Gy_gradient = torch.autograd.grad(inner_loss, self.alpha, create_graph=True)

        for g_grad, param in zip(Gy_gradient, self.alpha):
            G_gradient.append((param - self.args.hessian_lr * g_grad).view(-1))
        # G_gradient = torch.reshape(torch.hstack(G_gradient), [-1])

        for _ in range(self.args.hessian_q):
            # Jacobian = torch.matmul(G_gradient, v_0)
            v_new = torch.autograd.grad(G_gradient, self.alpha, grad_outputs=v_0, retain_graph=True)
            v_0 = v_new[0].data.detach()
            z_list.append(v_0)
        index = np.random.randint(self.args.hessian_q)
        v_Q = self.args.hessian_lr * z_list[index]
        # Gyx_gradient
        outputs = predict(self.inner_model, val_data)
        inner_loss = -self.aucloss(outputs, train_labels.to(self.device))
        Gy_gradient = torch.autograd.grad(inner_loss, self.alpha, retain_graph=True, create_graph=True)
        Gy_params = Gy_gradient[0]
        Gyxv_gradient = torch.autograd.grad(Gy_params, self.upper_variables, grad_outputs=v_Q, allow_unused=True)
        for i, (f_x, g_yxv) in enumerate(zip(Fx_gradient, Gyxv_gradient)):
            if g_yxv is not None:
                f_x.data -= g_yxv.data
            else:
                f_x.data -= torch.zeros(1).cuda()
        return Fx_gradient

def predict(net, inputs):
        """ Get predictions for a single batch. """
        s_embed, s_lens = inputs
        outputs = net((s_embed.cuda(), s_lens))
        return outputs






