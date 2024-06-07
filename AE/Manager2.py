import torch
from torch import nn, optim
from tqdm import trange, tqdm
from utils.metrics import AverageMeter,Metrics
from utils.functions import EarlyStopping
from data.base import DataManager
from AEModel import  AE
import argparse
import torch.nn.functional as F
import logging
import  os
import datetime

__all__ = ['Manager']

class Manager:

    def __init__(self, args, data, AEModel,FusionModel):

        self.logger = logging.getLogger(args.logger_name)
        self.device=args.device
        self.AEModel = AEModel
        self.FusionModel=FusionModel
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.FusionModel.parameters()), lr=args.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=args.gamma)

        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            data.mm_dataloader['train'], data.mm_dataloader['dev'], data.mm_dataloader['test']

        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.metrics = Metrics(args)
    def _train(self, args):
        early_stopping = EarlyStopping(args)
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.AEModel.eval()
            self.FusionModel.train()
            loss_record = AverageMeter()

            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):

                text_feats = batch['text_feats'].to(self.device)
                video_feats = batch['video_feats'].to(self.device)
                audio_feats = batch['audio_feats'].to(self.device)
                label_ids = batch['label_ids'].to(self.device)
                self.AEModel(text_feats, video_feats, audio_feats)
                with torch.set_grad_enabled(True):

                    outputs=self.FusionModel(self.AEModel.utt_private_t, self.AEModel.utt_private_v, self.AEModel.utt_private_a,
                           self.AEModel.utt_shared_t, self.AEModel.utt_shared_v, self.AEModel.utt_shared_a)

                    loss = self.criterion(outputs, label_ids)

                    self.optimizer.zero_grad()
                    loss.backward()
                    loss_record.update(loss.item(), label_ids.size(0))

                    if args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_([param for param in self.FusionModel.parameters() if param.requires_grad],
                                                  args.grad_clip)

                    self.optimizer.step()

            outputs = self._get_outputs(args, mode='test')
            eval_score = outputs[args.eval_monitor]

            early_stopping(eval_score, self.FusionModel)  # hxj
            eval_results = {
                'train_loss': round(loss_record.avg, 4),
                'best_eval_score': round(early_stopping.best_score, 4),
                'eval_score': round(eval_score, 4)
            }

            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(eval_results[key]))

            if early_stopping.counter != 0:
                self.lr_scheduler.step()

            if early_stopping.early_stop:
                self.logger.info(f'EarlyStopping at epoch {epoch + 1}')
                break

        self.best_eval_score = early_stopping.best_score
        self.model = early_stopping.best_model
        if args.save_model:
            print('Trained models are saved in %s', args.model_output_path)
            torch.save(self.FusionModel.state_dict(), 'fusion_model.bin')

    def _get_outputs(self, args, mode='eval', return_sample_results=False, show_results=True):

        if mode == 'eval':
            dataloader = self.eval_dataloader
        elif mode == 'test':
            dataloader = self.test_dataloader
        elif mode == 'train':
            dataloader = self.train_dataloader

        self.FusionModel.eval()
        self.AEModel.eval()

        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, args.num_labels)).to(self.device)

        loss_record = AverageMeter()

        for batch in tqdm(dataloader, desc="Iteration"):
            text_feats = batch['text_feats'].to(self.device)
            video_feats = batch['video_feats'].to(self.device)
            audio_feats = batch['audio_feats'].to(self.device)
            label_ids = batch['label_ids'].to(self.device)
            self.AEModel(text_feats, video_feats, audio_feats)
            with torch.set_grad_enabled(False):
                logits = self.FusionModel(self.AEModel.utt_private_t, self.AEModel.utt_private_v, self.AEModel.utt_private_a,
                           self.AEModel.utt_shared_t, self.AEModel.utt_shared_v, self.AEModel.utt_shared_a)
                total_logits = torch.cat((total_logits, logits))
                total_labels = torch.cat((total_labels, label_ids))

                loss = self.criterion(logits, label_ids)
                loss_record.update(loss.item(), label_ids.size(0))

        total_probs = F.softmax(total_logits.detach(), dim=1)
        total_maxprobs, total_preds = total_probs.max(dim=1)

        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()

        outputs = self.metrics(y_true, y_pred, show_results=show_results)
        outputs.update({'loss': loss_record.avg})

        if return_sample_results:
            outputs.update(
                {
                    'y_true': y_true,
                    'y_pred': y_pred
                }
            )

        return outputs
class Fusion(nn.Module):
    def __init__(self,args):
        super(Fusion, self).__init__()
        self.args=args
        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1',
                               nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size * 3))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(self.args.dropout_rate))
        self.fusion.add_module('fusion_layer_1_activation', nn.ReLU())
        self.fusion.add_module('fusion_layer_3',
                               nn.Linear(in_features=args.hidden_size * 3, out_features=args.num_labels))

        encoder_layer = nn.TransformerEncoderLayer(d_model=args.hidden_size, nhead=1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.gate_v = nn.Sequential()
        self.gate_v.add_module('linear', nn.Linear(in_features=args.hidden_size, out_features=1))
        self.gate_v.add_module('sigmoid', nn.Sigmoid())

        self.gate_a = nn.Sequential()
        self.gate_a.add_module('linear', nn.Linear(in_features=args.hidden_size, out_features=1))
        self.gate_a.add_module('sigmoid', nn.Sigmoid())

        encoder_layer1 = nn.TransformerEncoderLayer(d_model=args.hidden_size, nhead=1)
        self.transformer_encoder1 = nn.TransformerEncoder(encoder_layer1, num_layers=1)

        encoder_layer2 = nn.TransformerEncoderLayer(d_model=args.hidden_size, nhead=1)
        self.transformer_encoder2 = nn.TransformerEncoder(encoder_layer2, num_layers=1)

        encoder_layer3 = nn.TransformerEncoderLayer(d_model=args.hidden_size, nhead=1)
        self.transformer_encoder3 = nn.TransformerEncoder(encoder_layer3, num_layers=1)
    def forward(self,utt_private_t, utt_private_v, utt_private_a,
                          utt_shared_t, utt_shared_v, utt_shared_a):
        h_t = torch.stack((utt_private_t, utt_shared_t), dim=0)
        h_a = torch.stack((utt_private_a, utt_shared_a), dim=0)
        h_v = torch.stack((utt_private_v, utt_shared_v), dim=0)

        h_t = self.transformer_encoder1(h_t)
        h_t = torch.sum(h_t, dim=0, keepdim=False) / h_t.shape[0]

        h_a = self.transformer_encoder2(h_a)
        h_a = torch.sum(h_a, dim=0, keepdim=False) / h_a.shape[0]

        h_v = self.transformer_encoder3(h_v)
        h_v = torch.sum(h_v, dim=0, keepdim=False) / h_v.shape[0]

        t_a = self.transformer_encoder(torch.stack((h_t, h_a)))
        t_a = torch.sum(t_a, dim=0, keepdim=False) / t_a.shape[0]
        t_v = self.transformer_encoder(torch.stack((h_t, h_v)))
        t_v = torch.sum(t_v, dim=0, keepdim=False) / t_v.shape[0]

        gate_a = self.gate_a(t_a)
        gate_v = self.gate_v(t_v)

        # h = torch.cat((h_t,h_a*gate_a,h_v*gate_v), dim=1)
        h = h_t + h_a * gate_a + h_v * gate_v
        # h = torch.cat((self.utt_private_t, self.utt_private_v, self.utt_private_a, self.utt_shared_t, self.utt_shared_v, self.utt_shared_a), dim=1)
        # h = torch.cat((self.utt_private_t, self.utt_private_v, self.utt_private_a, self.utt_shared_t, self.utt_shared_v, self.utt_shared_a), dim=1)
        logits = self.fusion(h)
        return logits

def set_logger(args):
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    args.logger_name = f"{args.method}_{args.dataset}_{args.data_mode}_{time}"

    logger = logging.getLogger(args.logger_name)
    logger.setLevel(logging.DEBUG)

    log_path = os.path.join(args.log_path, args.logger_name + '.log')
    fh = logging.FileHandler(log_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(fh_formatter)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--logger_name', type=str, default='ae', help="Logger name for multimodal intent analysis.")

    parser.add_argument('--dataset', type=str, default='MIntRec', help="The name of the used dataset.")

    parser.add_argument('--data_mode', type=str, default='multi-class',
                        help="The task mode (multi-class or binary-class).")

    parser.add_argument('--method', type=str, default='ae', help="which method to use (text, mult, misa, mag_bert).")

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
    args.num_train_epochs=20
    args.use_cmd_sim=False
    args.padding_mode= 'zero'
    args.padding_loc= 'end'
    args.train = True
    args.data_mode = 'multi-class'
    args.text_backbone = 'bert-base-uncased'
    args.save_model = True
    args.eval_monitor='f1'
    args.hidden_size=256
    args.train_batch_size=6
    args.eval_batch_size=6
    args.test_batch_size=6
    args.wait_patience=4

    args.reverse_grad_weight=0.8
    args.hidden_size=256
    args.dropout_rate= 0.1
    args.diff_weight=0.7  # 0.7
    args.sim_weight= 0.7  # 0.7
    args.recon_weight= 0.6  # 0.6
    args.lr= 0.001
    args.grad_clip= -1.0
    args.gamma= 0.5
    args.model_output_path= '../results/'
    args.device=torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

    set_logger(args)
    #print(args)

    data = DataManager(args)
    ae=AE(args)
    ae.load_state_dict(torch.load('ae_model.bin'))
    ae.to(args.device)
    fusion=Fusion(args)
    fusion.to(args.device)
    manager=Manager(args,data,ae,fusion)
    manager._train(args)


