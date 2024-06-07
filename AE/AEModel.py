import torch
from torch import nn,optim
from torch.autograd import Function
from backbones.SubNets.FeatureNets import BERTEncoder
from torch.nn import  TransformerEncoderLayer
from torch.nn import TransformerEncoder as transE

class ReverseLayerF(Function):
    """
    Adapted from https://github.com/fungtion/DSN/blob/master/functions.py
    """
    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.p

        return output, None

class AE(nn.Module):
    
    def __init__(self, args):

        super(AE, self).__init__()

        self.text_subnet = BERTEncoder.from_pretrained(args.text_backbone, cache_dir = args.cache_path)
        self.visual_size = args.video_feat_dim
        self.acoustic_size = args.audio_feat_dim
        self.text_size = args.text_feat_dim

        self.dropout_rate = args.dropout_rate
        self.activation = nn.ReLU()
        
        self.input_sizes = input_sizes = [self.text_size, self.visual_size, self.acoustic_size]
        self.hidden_sizes = hidden_sizes = [int(self.text_size), int(self.visual_size), int(self.acoustic_size)]
        self.args = args


        self.transformer_encoder_layer_t = TransformerEncoderLayer(d_model=self.text_size, nhead=1,batch_first=True)
        self.transformer_encoder_t = transE(encoder_layer=self.transformer_encoder_layer_t, num_layers=1)

        self.transformer_encoder_layer_a = TransformerEncoderLayer(d_model=self.acoustic_size, nhead=1,batch_first=True)
        self.transformer_encoder_a = transE(encoder_layer=self.transformer_encoder_layer_a, num_layers=1)

        self.transformer_encoder_layer_v = TransformerEncoderLayer(d_model=self.visual_size, nhead=1,batch_first=True)
        self.transformer_encoder_v = transE(encoder_layer=self.transformer_encoder_layer_v, num_layers=1)




        self.project_t = nn.Sequential()
        self.project_t.add_module('project_t', nn.Linear(in_features=args.text_feat_dim, out_features=args.hidden_size))
        self.project_t.add_module('project_t_activation', self.activation)
        self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(args.hidden_size))
        
        self.project_v = nn.Sequential()
        self.project_v.add_module('project_v', nn.Linear(in_features=args.video_feat_dim, out_features=args.hidden_size))
        self.project_v.add_module('project_v_activation', self.activation)
        self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(args.hidden_size))

        self.project_a = nn.Sequential()
        self.project_a.add_module('project_a', nn.Linear(in_features=args.audio_feat_dim, out_features=args.hidden_size))
        self.project_a.add_module('project_a_activation', self.activation)
        self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(args.hidden_size))

        ##########################################
        # private encoders
        ##########################################
        self.private_t = nn.Sequential()
        self.private_t.add_module('private_t_1', nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size))
        self.private_t.add_module('private_t_activation_1', self.activation)
        
        self.private_v = nn.Sequential()
        self.private_v.add_module('private_v_1', nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size))
        self.private_v.add_module('private_v_activation_1', self.activation)

        self.private_a = nn.Sequential()
        self.private_a.add_module('private_a_3', nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size))
        self.private_a.add_module('private_a_activation_3', self.activation)

        ##########################################
        # shared encoder
        ##########################################
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())

        ##########################################
        # reconstruct
        ##########################################
        self.recon_t = nn.Sequential()
        self.recon_t.add_module('recon_t_1', nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size))
        self.recon_v = nn.Sequential()
        self.recon_v.add_module('recon_v_1', nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size))
        self.recon_a = nn.Sequential()
        self.recon_a.add_module('recon_a_1', nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size))

        if not args.use_cmd_sim:
            self.discriminator = nn.Sequential()
            self.discriminator.add_module('discriminator_layer_1', nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size))
            self.discriminator.add_module('discriminator_layer_1_activation', self.activation)
            self.discriminator.add_module('discriminator_layer_1_dropout', nn.Dropout(args.dropout_rate))
            self.discriminator.add_module('discriminator_layer_2', nn.Linear(in_features=args.hidden_size, out_features=len(hidden_sizes)))

        self.sp_discriminator = nn.Sequential()
        self.sp_discriminator.add_module('sp_discriminator_layer_1', nn.Linear(in_features=args.hidden_size, out_features = 4))


   
    def _reconstruct(self):
    
        self.utt_t = (self.utt_private_t + self.utt_shared_t)
        self.utt_v = (self.utt_private_v + self.utt_shared_v)
        self.utt_a = (self.utt_private_a + self.utt_shared_a)

        self.utt_t_recon = self.recon_t(self.utt_t)
        self.utt_v_recon = self.recon_v(self.utt_v)
        self.utt_a_recon = self.recon_a(self.utt_a)

    def _shared_private(self, utterance_t, utterance_v, utterance_a):
        
        # Projecting to same sized space
        self.utt_t_orig = utterance_t = self.project_t(utterance_t)
        self.utt_v_orig = utterance_v = self.project_v(utterance_v)
        self.utt_a_orig = utterance_a = self.project_a(utterance_a)

        # Private-shared components
        self.utt_private_t = self.private_t(utterance_t)
        self.utt_private_v = self.private_v(utterance_v)
        self.utt_private_a = self.private_a(utterance_a)

        self.utt_shared_t = self.shared(utterance_t)
        self.utt_shared_v = self.shared(utterance_v)
        self.utt_shared_a = self.shared(utterance_a)

    def forward(self, text_feats, video_feats, audio_feats):
        '''
        text_feats:[batch_size,3,max_length]
        video_feats_p:[batch_size,max_length,video_feat_dim]
        audio_feats_p:[batch_size,max_length,audio_feat_dim]
        '''

        video_feats, audio_feats = video_feats.float(), audio_feats.float()

        text_feats = self.text_subnet(text_feats)

        # extract features from text modality
        text_feats = self.transformer_encoder_t(text_feats)
        text_feats = torch.sum(text_feats, dim=1, keepdim=False) / text_feats.shape[1]

        # extract features from visual modality
        video_feats = self.transformer_encoder_v(video_feats)
        video_feats = torch.sum(video_feats, dim=1, keepdim=False) / video_feats.shape[1]
        # extract features from acoustic modality
        audio_feats = self.transformer_encoder_a(audio_feats)
        audio_feats = torch.sum(audio_feats, dim=1, keepdim=False) / audio_feats.shape[1]


        self._shared_private(text_feats, video_feats, audio_feats)
        
        if not self.args.use_cmd_sim:
            # discriminator
            reversed_shared_code_t = ReverseLayerF.apply(self.utt_shared_t, self.args.reverse_grad_weight)
            reversed_shared_code_v = ReverseLayerF.apply(self.utt_shared_v, self.args.reverse_grad_weight)
            reversed_shared_code_a = ReverseLayerF.apply(self.utt_shared_a, self.args.reverse_grad_weight)

            self.domain_label_t = self.discriminator(reversed_shared_code_t)
            self.domain_label_v = self.discriminator(reversed_shared_code_v)
            self.domain_label_a = self.discriminator(reversed_shared_code_a)
        else:
            self.domain_label_t = None
            self.domain_label_v = None
            self.domain_label_a = None

        self.shared_or_private_p_t = self.sp_discriminator(self.utt_private_t)
        self.shared_or_private_p_v = self.sp_discriminator(self.utt_private_v)
        self.shared_or_private_p_a = self.sp_discriminator(self.utt_private_a)
        self.shared_or_private_s = self.sp_discriminator((self.utt_shared_t + self.utt_shared_v + self.utt_shared_a) / 3.0)

        self._reconstruct()

       # return logits,gate_a,gate_v #,self.utt_private_t, self.utt_private_v, self.utt_private_a, self.utt_shared_t, self.utt_shared_v, self.utt_shared_a


