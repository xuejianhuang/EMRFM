import torch
from torch import nn, optim
from tqdm import trange, tqdm
from utils.functions import restore_model
from utils.metrics import AverageMeter
from util import MSE, CMD, DiffLoss
from data.base import DataManager
from AEModel import  AE
import argparse

__all__ = ['Manager']

class Manager:

    def __init__(self, args, data, model):


        self.device=args.device
        self.model = model

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=args.gamma)

        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            data.mm_dataloader['train'], data.mm_dataloader['dev'], data.mm_dataloader['test']

        self.args = args
        self.domain_loss_criterion = nn.CrossEntropyLoss(reduction='mean')
        self.criterion = nn.CrossEntropyLoss()
        self.loss_diff = DiffLoss()
        self.loss_recon = MSE()
        self.loss_cmd = CMD()

        if args.train:
            self.best_eval_score = 0
        else:
            self.model.load_state_dict(torch.load('ae_model.bin'))

    def _train(self, args):

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            loss_record = AverageMeter()

            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):

                text_feats = batch['text_feats'].to(self.device)
                video_feats = batch['video_feats'].to(self.device)
                audio_feats = batch['audio_feats'].to(self.device)
                label_ids = batch['label_ids'].to(self.device)

                with torch.set_grad_enabled(True):

                    self.model(text_feats, video_feats, audio_feats)

                    diff_loss = self._get_diff_loss()
                    domain_loss = self._get_domain_loss()
                    recon_loss = self._get_recon_loss()
                    cmd_loss = self._get_cmd_loss()

                    if self.args.use_cmd_sim:
                        similarity_loss = cmd_loss
                    else:
                        similarity_loss = domain_loss

                    loss =args.diff_weight * diff_loss + args.sim_weight * similarity_loss + args.recon_weight * recon_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    loss_record.update(loss.item(), label_ids.size(0))

                    if args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad],
                                                  args.grad_clip)

                    self.optimizer.step()
            print("train losss:",loss_record.avg)

        if args.save_model:
            print('Trained models are saved')
            torch.save(self.model.state_dict(), 'ae_model.bin')


    def _get_domain_loss(self):

        if self.args.use_cmd_sim:
            return 0.0

        # Predicted domain labels
        domain_pred_t = self.model.domain_label_t
        domain_pred_v = self.model.domain_label_v
        domain_pred_a = self.model.domain_label_a

        # True domain labels
        domain_true_t = torch.LongTensor([0] * domain_pred_t.size(0)).to(self.device)
        domain_true_v = torch.LongTensor([1] * domain_pred_v.size(0)).to(self.device)
        domain_true_a = torch.LongTensor([2] * domain_pred_a.size(0)).to(self.device)

        # Stack up predictions and true labels
        domain_pred = torch.cat((domain_pred_t, domain_pred_v, domain_pred_a), dim=0)
        domain_true = torch.cat((domain_true_t, domain_true_v, domain_true_a), dim=0)

        return self.domain_loss_criterion(domain_pred, domain_true)

    def _get_cmd_loss(self):

        if not self.args.use_cmd_sim:
            return 0.0

        # losses between shared states
        loss = self.loss_cmd(self.model.utt_shared_t, self.model.utt_shared_v, 5)
        loss += self.loss_cmd(self.model.utt_shared_t, self.model.utt_shared_a, 5)
        loss += self.loss_cmd(self.model.utt_shared_a, self.model.utt_shared_v, 5)
        loss = loss / 3.0

        return loss

    def _get_diff_loss(self):

        shared_t = self.model.utt_shared_t
        shared_v = self.model.utt_shared_v
        shared_a = self.model.utt_shared_a
        private_t = self.model.utt_private_t
        private_v = self.model.utt_private_v
        private_a = self.model.utt_private_a

        # Between private and shared
        loss = self.loss_diff(private_t, shared_t)
        loss += self.loss_diff(private_v, shared_v)
        loss += self.loss_diff(private_a, shared_a)

        # Across privates
        loss += self.loss_diff(private_a, private_t)
        loss += self.loss_diff(private_a, private_v)
        loss += self.loss_diff(private_t, private_v)

        return loss / 6.0

    def _get_recon_loss(self, ):

        loss = self.loss_recon(self.model.utt_t_recon, self.model.utt_t_orig)
        loss += self.loss_recon(self.model.utt_v_recon, self.model.utt_v_orig)
        loss += self.loss_recon(self.model.utt_a_recon, self.model.utt_a_orig)
        loss = loss / 3.0
        return loss



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--logger_name', type=str, default='ae', help="Logger name for multimodal intent analysis.")

    parser.add_argument('--dataset', type=str, default='MIntRec', help="The name of the used dataset.")

    parser.add_argument('--data_mode', type=str, default='multi-class',
                        help="The task mode (multi-class or binary-class).")

    parser.add_argument('--method', type=str, default='text', help="which method to use (text, mult, misa, mag_bert).")

    parser.add_argument("--text_backbone", type=str, default='bert',
                        help="which backbone to use for the text modality.")

    parser.add_argument('--seed', type=int, default=0, help="The random seed for initialization.")

    parser.add_argument('--num_workers', type=int, default=0, help="The number of workers to load data.")

    parser.add_argument('--gpu_id', type=str, default='0', help="The used gpu index of your device.")

    parser.add_argument("--data_path", default='..\\MIA-Datasets', type=str,
                        help="The input data dir. Should contain text, video and audio data for the task.")

    parser.add_argument("--train", action="store_true",
                        help="Whether to train the model.")  # 默认为False,要运行时该变量有传参就将为True

    parser.add_argument("--tune", action="store_true",
                        help="Whether to tune the model with a series of hyper-parameters.")

    parser.add_argument("--save_model", action="store_true",
                        help="whether to save trained-model for multimodal intent recognition.")

    parser.add_argument("--save_results", action="store_true",
                        help="whether to save final results for multimodal intent recognition.")

    parser.add_argument('--log_path', type=str, default='..\\logs', help="Logger directory.")

    parser.add_argument('--cache_path', type=str, default='..\\cache', help="The caching directory for pre-trained models.")

    parser.add_argument('--video_data_path', type=str, default='video_data', help="The directory of the video data.")

    parser.add_argument('--audio_data_path', type=str, default='audio_data', help="The directory of the audio data.")

    parser.add_argument('--video_feats_path', type=str, default='video_feats.pkl',
                        help="The directory of the video features.")

    parser.add_argument('--audio_feats_path', type=str, default='audio_feats.pkl',
                        help="The directory of the audio features.")

    parser.add_argument('--results_path', type=str, default='results', help="The path to save results.")

    parser.add_argument("--output_path", default='outputs', type=str,
                        help="The output directory where all train data will be written.")

    parser.add_argument("--model_path", default='models', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--config_file_name", type=str, default='text_bert_tune(.py)',
                        help="The name of the config file.")

    parser.add_argument("--results_file_name", type=str, default='results.csv',
                        help="The file name of all the experimental results.")


    args = parser.parse_args()
    args.num_train_epochs=1
    args.use_cmd_sim=False
    args.padding_mode= 'zero'
    args.padding_loc= 'end'
    args.train = True
    args.data_mode = 'multi-class'
    args.text_backbone = 'bert-base-uncased'
    args.save_model = True
    args.hidden_size=256
    args.train_batch_size=6
    args.eval_batch_size=6
    args.test_batch_size=6

    args.reverse_grad_weight=0.8
    args.hidden_size=256
    args.dropout_rate= 0
    args.diff_weight=0.7  # 0.7
    args.sim_weight= 0.7  # 0.7
    args.recon_weight= 0.6  # 0.6
    args.lr= 0.00001
    args.grad_clip= -1.0
    args.gamma= 0.5
    args.model_output_path= '../results/'
    args.device=torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'


    #print(args)

    data = DataManager(args)
    ae=AE(args)
    ae.to(args.device)
    manager=Manager(args,data,ae)
    manager._train(args)


