import torch
import torch.nn as nn
from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs
from utils import clones

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
        out = torch.mean(out, dim=1)
        #(batch, out_dim)
        
        #out = self.layer5(out)
        #(Batch, num_classes)
        return out

class BasicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size, bias=False, **kwargs)

    def forward(self, x):
        ret = self.conv(x)
        return ret


class TemporalFeatureAggregator(nn.Module):
    def __init__(self, in_channels, squeeze=4, parts_num=16):
        super(TemporalFeatureAggregator, self).__init__()
        hidden_dim = int(in_channels // squeeze)
        self.parts_num = parts_num

        # MTB1
        conv3x1 = nn.Sequential(
            BasicConv1d(in_channels, hidden_dim, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            BasicConv1d(hidden_dim, in_channels, 1))
        self.conv1d3x1 = clones(conv3x1, parts_num)
        self.avg_pool3x1 = nn.AvgPool1d(3, stride=1, padding=1)
        self.max_pool3x1 = nn.MaxPool1d(3, stride=1, padding=1)

        # MTB1
        conv3x3 = nn.Sequential(
            BasicConv1d(in_channels, hidden_dim, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            BasicConv1d(hidden_dim, in_channels, 3, padding=1))
        self.conv1d3x3 = clones(conv3x3, parts_num)
        self.avg_pool3x3 = nn.AvgPool1d(5, stride=1, padding=2)
        self.max_pool3x3 = nn.MaxPool1d(5, stride=1, padding=2)

        # Temporal Pooling, TP
        self.TP = torch.max

    def forward(self, x):
        """
          Input:  x,   [n, c, s, p]
          Output: ret, [n, c, p]
        """
        n, c, s, p = x.size()
        x = x.permute(3, 0, 1, 2).contiguous()  # [p, n, c, s]
        feature = x.split(1, 0)  # [[1, n, c, s], ...]
        x = x.view(-1, c, s)

        # MTB1: ConvNet1d & Sigmoid
        logits3x1 = torch.cat([conv(_.squeeze(0)).unsqueeze(0)
                               for conv, _ in zip(self.conv1d3x1, feature)], 0)
        scores3x1 = torch.sigmoid(logits3x1)
        # MTB1: Template Function
        feature3x1 = self.avg_pool3x1(x) + self.max_pool3x1(x)
        feature3x1 = feature3x1.view(p, n, c, s)
        feature3x1 = feature3x1 * scores3x1

        # MTB2: ConvNet1d & Sigmoid
        logits3x3 = torch.cat([conv(_.squeeze(0)).unsqueeze(0)
                               for conv, _ in zip(self.conv1d3x3, feature)], 0)
        scores3x3 = torch.sigmoid(logits3x3)
        # MTB2: Template Function
        feature3x3 = self.avg_pool3x3(x) + self.max_pool3x3(x)
        feature3x3 = feature3x3.view(p, n, c, s)
        feature3x3 = feature3x3 * scores3x3

        # Temporal Pooling
        ret = self.TP(feature3x1 + feature3x3, dim=-1)[0]  # [p, n, c]
        ret = ret.permute(1, 2, 0).contiguous()  # [n, p, c]
        return ret


class GaitPart_occ_aware(BaseModel):
    def __init__(self, *args, **kargs):
        super(GaitPart_occ_aware, self).__init__(*args, **kargs)
        """
            GaitPart: Temporal Part-based Model for Gait Recognition
            Paper:    https://openaccess.thecvf.com/content_CVPR_2020/papers/Fan_GaitPart_Temporal_Part-Based_Model_for_Gait_Recognition_CVPR_2020_paper.pdf
            Github:   https://github.com/ChaoFan96/GaitPart
        """

    def build_network(self, model_cfg):
        
        self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        head_cfg = model_cfg['SeparateFCs']
        occ_dim = model_cfg['occ_dim']
        self.parts_num = model_cfg['SeparateFCs']['parts_num']
        
        self.Head = SeparateFCs(model_cfg['SeparateFCs']['parts_num'], occ_dim + model_cfg['SeparateFCs']['in_channels'], occ_dim + model_cfg['SeparateFCs']['out_channels'])
        self.Backbone = SetBlockWrapper(self.Backbone)
        self.HPP = SetBlockWrapper(
            HorizontalPoolingPyramid(bin_num=model_cfg['bin_num']))
        self.TFA = PackSequenceWrapper(TemporalFeatureAggregator(
            in_channels=head_cfg['in_channels'], parts_num=head_cfg['parts_num']))
        
        self.occ_detector = Occ_Detector()
        self.occ_detector.eval()
        self.occ_detector.requires_grad_(False)
        self.occ_mixer_fc = nn.Sequential(
            SeparateFCs(model_cfg['SeparateFCs']['parts_num'], occ_dim + model_cfg['SeparateFCs']['in_channels'], occ_dim + model_cfg['SeparateFCs']['out_channels']),     #(parts_num, original_gaitgl embed size, desired size of occlusion aware embedding)
            nn.LeakyReLU(inplace=True)
        )

    def init_parameters(self):
        super(GaitPart_occ_aware, self).init_parameters()
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

        sils = ipts[0]
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)

        del ipts
        
        with torch.no_grad():
            occ_embed = self.occ_detector(sils)     #(Batch, embed)
            occ_embed = occ_embed.unsqueeze(2).repeat(1,1,self.parts_num)   #(batch, embed, p)  repeated
        
        out = self.Backbone(sils)  # [n, c, s, h, w]
        out = self.HPP(out)  # [n, c, s, p]
        out = self.TFA(out, seqL)  # [n, c, p]

        out = torch.cat([out, occ_embed], dim=1)  #(n, c + occ_dim, p)
        out = self.occ_mixer_fc(out)  #(n, new_dim_size, p)
        #print(f"Occ awareness added")
        embs = self.Head(out)  # [n, c, p]
        
        n, c, s, h, w = sils.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embs, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': sils.reshape(n*s, c, h, w)
            },
            'inference_feat': {
                'embeddings': embs
            }
        }
        return retval
