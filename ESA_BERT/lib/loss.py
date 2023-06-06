import torch
import torch.nn as nn
from torch.autograd import Variable


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss (max-margin based)
    """

    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def max_violation_on(self):
        self.max_violation = True
        print('Use VSE++ objective.')

    def max_violation_off(self):
        self.max_violation = False
        print('Use VSE0 objective.')

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = get_sim(im, s)#bk,b
        hardnum = self.opt.hardnum
        mm_a = (torch.arange(scores.size(0))//self.opt.hardnum +1)*self.opt.hardnum
        mask_a = torch.arange(im.size(0)).view(im.size(0),1).expand_as(scores)
        mask1 = (mask_a<mm_a.long())
        mask = mask1 * mask1.t()
        if torch.cuda.is_available():
            I = mask.cuda()

        #caption retrieval 
        scores_inner = torch.masked_select(scores,I).reshape(scores.size(0)//hardnum, hardnum, hardnum)
        
        scores_image = scores_inner.min(dim=2)[0].reshape((-1,1))
        cost_s = (self.margin + scores - scores_image.view(im.size(0),1).expand_as(scores)).clamp(min=0)

        #image retrieval
        scores_caption = scores_inner.min(dim=1)[0].reshape((1,-1))
        cost_im = (self.margin + scores - scores_caption.view(1,s.size(0)).expand_as(scores)).clamp(min=0)

        
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        if self.max_violation:
            cost_im = cost_im.max(0)[0]
            cost_s = cost_s.max(1)[0]
            
        cost_im =cost_im.sum()    
        cost_s =cost_s.sum()

        return cost_im, cost_s

def get_sim(images, captions):
    similarities = images.mm(captions.t())
    return similarities

