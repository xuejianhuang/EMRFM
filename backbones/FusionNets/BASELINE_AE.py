import torch
from torch import nn
from ..SubNets.FeatureNets import BERTEncoder
from torch.nn import  TransformerEncoderLayer, TransformerEncoder

__all__ = ['BASELINE']

class BASELINE(nn.Module):
    
    def __init__(self, args):

        super(BASELINE, self).__init__()

        self.text_subnet = BERTEncoder.from_pretrained(args.text_backbone, cache_dir = args.cache_path)
        self.visual_size = args.video_feat_dim
        self.acoustic_size = args.audio_feat_dim
        self.text_size = args.text_feat_dim

        self.dropout_rate = args.dropout_rate
        self.output_dim = args.num_labels
        self.activation = nn.ReLU()
        self.tanh = nn.Tanh()
        
        self.input_sizes = input_sizes = [self.text_size, self.visual_size, self.acoustic_size]
        self.args = args


        self.transformer_encoder_layer_t = TransformerEncoderLayer(d_model=self.text_size, nhead=1,batch_first=True)
        self.transformer_encoder_t = TransformerEncoder(encoder_layer=self.transformer_encoder_layer_t, num_layers=1)

        self.transformer_encoder_layer_a = TransformerEncoderLayer(d_model=self.acoustic_size, nhead=1,batch_first=True)
        self.transformer_encoder_a = TransformerEncoder(encoder_layer=self.transformer_encoder_layer_a, num_layers=1)

        self.transformer_encoder_layer_v = TransformerEncoderLayer(d_model=self.visual_size, nhead=1,batch_first=True)
        self.transformer_encoder_v = TransformerEncoder(encoder_layer=self.transformer_encoder_layer_v, num_layers=1)


        encoder_layer= nn.TransformerEncoderLayer(d_model=args.hidden_size, nhead=1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.gate_v = nn.Sequential()
        self.gate_v.add_module('linear', nn.Linear(in_features=args.hidden_size, out_features=1))
        self.gate_v.add_module('sigmoid', nn.Sigmoid())

        self.gate_a = nn.Sequential()
        self.gate_a.add_module('linear', nn.Linear(in_features=args.hidden_size, out_features=1))
        self.gate_a.add_module('sigmoid', nn.Sigmoid())


        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size*3))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(self.dropout_rate))
        self.fusion.add_module('fusion_layer_1_activation', nn.ReLU())
        self.fusion.add_module('fusion_layer_3', nn.Linear(in_features=args.hidden_size*3, out_features= self.output_dim))

        self.project_t = nn.Sequential()
        self.project_t.add_module('project_t', nn.Linear(in_features=self.text_size, out_features=args.hidden_size))
        self.project_t.add_module('project_t_activation', nn.ReLU())
        self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(args.hidden_size))

        self.project_v = nn.Sequential()
        self.project_v.add_module('project_v', nn.Linear(in_features=self.visual_size, out_features=args.hidden_size))
        self.project_v.add_module('project_v_activation', nn.ReLU())
        self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(args.hidden_size))

        self.project_a = nn.Sequential()
        self.project_a.add_module('project_a', nn.Linear(in_features=self.acoustic_size, out_features=args.hidden_size))
        self.project_a.add_module('project_a_activation', self.activation)
        self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(args.hidden_size))

    def forward(self, text_feats, video_feats, audio_feats):
        '''
        text_feats:[batch_size,3,max_length]
        video_feats_p:[batch_size,max_length,video_feat_dim]
        audio_feats_p:[batch_size,max_length,audio_feat_dim]
        '''

        text_feats = self.text_subnet(text_feats)

        #extract features from text modality
        text_feats = self.transformer_encoder_t(text_feats)
        text_feats = torch.sum(text_feats, dim=1, keepdim=False) / text_feats.shape[1]

        text_feats=self.project_t(text_feats)

        #extract features from visual modality
        video_feats = self.transformer_encoder_v(video_feats)
        video_feats = torch.sum(video_feats, dim=1, keepdim=False) / video_feats.shape[1]

        video_feats=self.project_v(video_feats)

        # extract features from acoustic modality
        audio_feats = self.transformer_encoder_a(audio_feats)
        audio_feats = torch.sum(audio_feats, dim=1, keepdim=False) / audio_feats.shape[1]

        audio_feats=self.project_a(audio_feats)




        t_a = self.transformer_encoder(torch.stack((text_feats, audio_feats)))
        t_a = torch.sum(t_a, dim=0, keepdim=False) / t_a.shape[0]
        t_v = self.transformer_encoder(torch.stack((text_feats, video_feats)))
        t_v = torch.sum(t_v, dim=0, keepdim=False) / t_v.shape[0]

        gate_a = self.gate_a(t_a)
        gate_v = self.gate_v(t_v)


        h=text_feats+audio_feats*gate_a+video_feats*gate_v
        #h = torch.cat((self.utt_private_t, self.utt_private_v, self.utt_private_a, self.utt_shared_t, self.utt_shared_v, self.utt_shared_a), dim=1)
        #h = torch.cat((self.utt_private_t, self.utt_private_v, self.utt_private_a, self.utt_shared_t, self.utt_shared_v, self.utt_shared_a), dim=1)
        logits = self.fusion(h)
        return logits