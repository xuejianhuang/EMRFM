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


        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=self.text_size+self.acoustic_size+self.visual_size, out_features=args.hidden_size*3))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(self.dropout_rate))
        self.fusion.add_module('fusion_layer_1_activation', self.activation)
        self.fusion.add_module('fusion_layer_3', nn.Linear(in_features=args.hidden_size*3, out_features= self.output_dim))

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

        # extract features from visual modality
        video_feats = self.transformer_encoder_v(video_feats)
        video_feats = torch.sum(video_feats, dim=1, keepdim=False) / video_feats.shape[1]

        # extract features from acoustic modality
        audio_feats = self.transformer_encoder_a(audio_feats)
        audio_feats = torch.sum(audio_feats, dim=1, keepdim=False) / audio_feats.shape[1]


        h = torch.cat((text_feats,audio_feats,video_feats), dim=1)
       # h=h_t+h_a*gate_a+h_v*gate_v
        #h = torch.cat((self.utt_private_t, self.utt_private_v, self.utt_private_a, self.utt_shared_t, self.utt_shared_v, self.utt_shared_a), dim=1)
        #h = torch.cat((self.utt_private_t, self.utt_private_v, self.utt_private_a, self.utt_shared_t, self.utt_shared_v, self.utt_shared_a), dim=1)
        logits = self.fusion(h)
        return logits