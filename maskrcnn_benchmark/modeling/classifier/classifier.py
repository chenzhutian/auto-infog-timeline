import torch
from torch import nn
import torch.nn.functional as F

from maskrcnn_benchmark.modeling.utils import cat

class ClassifierModule(nn.Module):

    def __init__(self, field, loss_weighted, cfg):
        super(ClassifierModule, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        num_classes = (cfg.MODEL.CLASSIFIER_CLS 
                        if field == 'image_class' 
                        else cfg.MODEL.CLASSIFIER_ORIENT).NUM_CLASSES

        fpn_out_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.fc = nn.Linear(fpn_out_channels, num_classes)

        nn.init.normal_(self.fc.weight, mean=0, std=0.01)
        nn.init.constant_(self.fc.bias, 0)

        self.field = field
        self.loss_weighted = loss_weighted
    
    def forward(self, x, targets = None):
        ''' 
        Arguments:
            x (Tensor) B x C x W x H
            targets (list[BoxList]) targets of each image
        
        Returns:
            classification_loss (Tensor)
        '''
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if not self.training:
            class_prob, class_label = torch.max(F.softmax(x, dim=1), 1)
            # x: BxC, class_label: (Bx1, Bx1)
            return x, (class_prob, class_label), {}

        # B x 1
        # labels = torch.stack([target.get_field("image_class") for target in targets])
        labels = torch.stack([target.get_field(self.field) for target in targets])
        classification_loss = F.cross_entropy(x, labels)

        # x: fc features, labels: tensor, dict: dict of loss
        # PS: weighted the loss to make it in the same scale of other loss
        # For more information please refer to https://github.com/chenzhutian/2019-infovis-arposter-maskrcnn-benchmark/issues/1
        loss = dict(loss_image_classifier=classification_loss * self.loss_weighted) if self.field == 'image_class' else dict(loss_image_orientation_cls=classification_loss * self.loss_weighted)

        return x, labels, loss

def build_classifier(field, loss_weighted, cfg):
    return ClassifierModule(field, loss_weighted, cfg)
