import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel
from ..modules import SeparateFCs, BasicConv3d, PackSequenceWrapper, SeparateBNNecks


class Occ_Detector(torch.nn.Module):

    def __init__(self):
        super(Occ_Detector, self).__init__()
        
        keep_prob = 1
        # L1 ImgIn shape=(?, 64, 64, 1)
        # Conv -> (?, 64, 64, 32)
        # Pool -> (?, 32, 32, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - keep_prob))
        # L2 ImgIn shape=(?, 32, 32, 32)
        # Conv      ->(?, 32, 32, 64)
        # Pool      ->(?, 16, 16, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - keep_prob))
        # L3 ImgIn shape=(?, 16, 16, 64)
        # Conv ->(?, 16,16, 128)
        # Pool ->(?, 8, 8, 128)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            torch.nn.Dropout(p=1 - keep_prob))

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        
        
        # L4 FC 4x4x128 inputs -> 625 outputs
        self.fc1 = torch.nn.Linear(128, 64, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - keep_prob))
        # L5 Final FC 625 inputs -> 10 outputs
        # self.fc2 = torch.nn.Linear(64, 9, bias=True)
        # torch.nn.init.xavier_uniform_(self.fc2.weight) # initialize parameters
        # self.layer5 = torch.nn.Sequential(
        #     self.fc2, 
        #     torch.nn.Softmax(dim=1)
        # )
        return 
    
    def forward(self, x):
        #x: (Batch, frames, c, h, w)    #(c = 1)
        b, c, f, h, w = x.shape
        x = x.permute(0,2,1,3,4).contiguous()    #(b, f, c, h, w)
        
        out = x.view(b*f, c, h, w)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        
        out = out.view(out.size(0), -1)   # Flatten them for FC
        out = self.layer4(out)
        #(Batch*f, out_dim)
        
        out = out.view(b, f, -1)
        #(batch, frames, channels)
        
        out = out.permute(0,2,1).contiguous()
        #(batch, channels, frames)
        
        return out    

class GLConv(nn.Module):
    def __init__(self, in_channels, out_channels, halving, fm_sign=False, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, **kwargs):
        super(GLConv, self).__init__()
        self.halving = halving
        self.fm_sign = fm_sign
        self.global_conv3d = BasicConv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)
        self.local_conv3d = BasicConv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)

    def forward(self, x):
        '''
            x: [n, c, s, h, w]
        '''
        gob_feat = self.global_conv3d(x)
        if self.halving == 0:
            lcl_feat = self.local_conv3d(x)
        else:
            h = x.size(3)
            split_size = int(h // 2**self.halving)
            lcl_feat = x.split(split_size, 3)
            lcl_feat = torch.cat([self.local_conv3d(_) for _ in lcl_feat], 3)

        if not self.fm_sign:
            feat = F.leaky_relu(gob_feat) + F.leaky_relu(lcl_feat)
        else:
            feat = F.leaky_relu(torch.cat([gob_feat, lcl_feat], dim=3))
        return feat


class GeMHPP(nn.Module):
    def __init__(self, bin_num=[64], p=6.5, eps=1.0e-6):
        super(GeMHPP, self).__init__()
        self.bin_num = bin_num
        self.p = nn.Parameter(
            torch.ones(1)*p)
        self.eps = eps

    def gem(self, ipts):
        return F.avg_pool2d(ipts.clamp(min=self.eps).pow(self.p), (1, ipts.size(-1))).pow(1. / self.p)

    def forward(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p] 
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = self.gem(z).squeeze(-1)
            features.append(z)
        return torch.cat(features, -1)


class GaitGL_occ_aware_guided_lin(BaseModel):
    """
        GaitGL: Gait Recognition via Effective Global-Local Feature Representation and Local Temporal Aggregation
        Arxiv : https://arxiv.org/pdf/2011.01461.pdf
    """

    def __init__(self, *args, **kargs):
        super(GaitGL_occ_aware_guided_lin, self).__init__(*args, **kargs)

    def build_network(self, model_cfg):
        in_c = model_cfg['channels']
        class_num = model_cfg['class_num']
        dataset_name = self.cfgs['data_cfg']['dataset_name']
        occ_dim = model_cfg['occ_dim']
        
        
        if dataset_name in ['OUMVLP', 'GREW']:
            # For OUMVLP and GREW
            self.conv3d = nn.Sequential(
                BasicConv3d(1, in_c[0], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True),
                BasicConv3d(in_c[0], in_c[0], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True),
            )
            self.LTA = nn.Sequential(
                BasicConv3d(in_c[0], in_c[0], kernel_size=(
                    3, 1, 1), stride=(3, 1, 1), padding=(0, 0, 0)),
                nn.LeakyReLU(inplace=True)
            )
            
            self.occ_aware = nn.Sequential(
                BasicConv3d(occ_dim + in_c[0], in_c[0], kernel_size=(
                    1,1,1), stride=(1, 1, 1), padding=(0, 0, 0)),
                nn.LeakyReLU(inplace=True)
            )

            self.GLConvA0 = nn.Sequential(
                GLConv(in_c[0], in_c[1], halving=1, fm_sign=False, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                GLConv(in_c[1], in_c[1], halving=1, fm_sign=False, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            )
            self.MaxPool0 = nn.MaxPool3d(
                kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.GLConvA1 = nn.Sequential(
                GLConv(in_c[1], in_c[2], halving=1, fm_sign=False, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                GLConv(in_c[2], in_c[2], halving=1, fm_sign=False, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            )
            self.GLConvB2 = nn.Sequential(
                GLConv(in_c[2], in_c[3], halving=1, fm_sign=False,  kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                GLConv(in_c[3], in_c[3], halving=1, fm_sign=True,  kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            )
        else:
            # For CASIA-B or other unstated datasets.
            self.conv3d = nn.Sequential(
                BasicConv3d(1, in_c[0], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True)
            )
            self.LTA = nn.Sequential(
                BasicConv3d(in_c[0], in_c[0], kernel_size=(
                    3, 1, 1), stride=(3, 1, 1), padding=(0, 0, 0)),
                nn.LeakyReLU(inplace=True)
            )

            self.GLConvA0 = GLConv(in_c[0], in_c[1], halving=3, fm_sign=False, kernel_size=(
                3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            self.MaxPool0 = nn.MaxPool3d(
                kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.GLConvA1 = GLConv(in_c[1], in_c[2], halving=3, fm_sign=False, kernel_size=(
                3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            self.GLConvB2 = GLConv(in_c[2], in_c[2], halving=3, fm_sign=True,  kernel_size=(
                3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = GeMHPP()

        self.Head0 = SeparateFCs(64, in_c[-1], in_c[-1])

        self.occ_detector = Occ_Detector()
        self.occ_detector.eval()
        self.occ_detector.requires_grad_(False)
        self.occ_mixer_fc = SeparateFCs(64, occ_dim + in_c[-1], occ_dim + in_c[-1])     #(parts_num, original_gaitgl embed size, desired size of occlusion aware embedding)
        
        if 'SeparateBNNecks' in model_cfg.keys():
            self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
            self.Bn_head = False
        else:
            self.Bn = nn.BatchNorm1d(occ_dim + in_c[-1])
            self.Head1 = SeparateFCs(64, occ_dim + in_c[-1], class_num)         
            self.Bn_head = True
            
            
    def init_parameters(self):
        super(GaitGL_occ_aware_guided_lin, self).init_parameters()
        if 'occ_detector_path' in self.cfgs['model_cfg']:
            model_path = self.cfgs['model_cfg']['occ_detector_path']
            pretrained_model = torch.load(model_path)
            
            new_dict = {}
            for k,v in pretrained_model.items():
                if k.split('.')[1] in ['fc2', 'layer5']:
                    continue    #Skip these last layers.
                else:
                    new_key = '.'.join(k.split('.')[1:])
                    new_dict[new_key] = v
            
            self.occ_detector.load_state_dict(new_dict)
            if torch.distributed.get_rank() == 0:
                print(f"\nOCCLUSION DETECTOR LOADED FROM: {model_path}\n")
        else:
            raise ValueError("Specify occ_detector_path in model_cfg!")
                
            

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        seqL = None if not self.training else seqL
        if not self.training and len(labs) != 1:
            raise ValueError(
                'The input size of each GPU must be 1 in testing mode, but got {}!'.format(len(labs)))
        sils = ipts[0].unsqueeze(1)
        del ipts
        n, _, s, h, w = sils.size()     #(batch, channel, frames, height, width)
        
        
        if s < 3:
            repeat = 3 if s == 1 else 2
            sils = sils.repeat(1, 1, repeat, 1, 1)

        outs = self.conv3d(sils)
        
        #pdb.set_trace()
        with torch.no_grad():
            _,_,_, h_out, w_out = outs.shape
            occ_embed = self.occ_detector(sils)     #(Batch, embed)
            
        occ_embed_mean = torch.mean(occ_embed, dim=2)   #(Batch, embed)
        occ_embed_mean = occ_embed_mean.unsqueeze(2).repeat(1,1,64)   #(batch, embed, p)  repeated
        
            
        occ_embed = occ_embed.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,h_out,w_out)  
        outs = torch.cat([outs, occ_embed], dim=1)   #(batch, channels + occ_dim, frames, h, w)
        outs = self.occ_aware(outs)    #(batch, channels, frames, h, w)
        #print(f"3d conv done")
        
        outs = self.LTA(outs)

        outs = self.GLConvA0(outs)
        outs = self.MaxPool0(outs)

        outs = self.GLConvA1(outs)
        outs = self.GLConvB2(outs)  # [n, c, s, h, w]

        outs = self.TP(outs, seqL=seqL, options={"dim": 2})[0]  # [n, c, h, w]
        outs = self.HPP(outs)  # [n, c, p]

        gait = self.Head0(outs)  # [n, c, p]

        gait = torch.cat([gait, occ_embed_mean], dim=1)  #(n, c + occ_dim, p)
        gait = self.occ_mixer_fc(gait)  #(n, new_dim_size, p)
        #print(gait.shape)
        #print(f"Lin done")
        
        
        if self.Bn_head:  # Original GaitGL Head
            bnft = self.Bn(gait)  # [n, 2*c, p]
            logi = self.Head1(bnft)  # bnft is [n, 64+c, p]
            embed = bnft
        else:  # BNNechk as Head
            bnft, logi = self.BNNecks(gait)  # [n, c, p]
            embed = gait
            
        
        n, c, s, h, w = sils.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed, 'labels': labs},
                'softmax': {'logits': logi, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': sils.reshape(n*s, c, h, w)
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval
