# http://ni4muraano.hatenablog.com/entry/2020/01/13/161152
# https://towardsdatascience.com/the-unknown-benefits-of-using-a-soft-f1-loss-in-classification-systems-753902c0105d

class MacroSoftF1Loss(nn.Module):
    def __init__(self, consider_true_negative=True, sigmoid_is_applied_to_input=False):
        super(MacroSoftF1Loss, self).__init__()
        self._consider_true_negative = consider_true_negative
        self._sigmoid_is_applied_to_input = sigmoid_is_applied_to_input

    def forward(self, input_, target):
        target = target.float()
        if not self._sigmoid_is_applied_to_input:
            input_ = torch.sigmoid(input_)

        TP = torch.sum(input_ * target, dim=0)
        FP = torch.sum((1 - input_) * target, dim=0)
        FN = torch.sum(input_ * (1 - target), dim=0)
        F1_class1 = 2 * TP / (2 * TP + FP + FN + 1e-8)
        loss_class1 = 1 - F1_class1

        if self._consider_true_negative:
            TN = torch.sum((1 - input_) * (1 - target), dim=0)
            F1_class0 = 2*TN/(2*TN + FP + FN + 1e-8)
            loss_class0 = 1 - F1_class0
            loss = (loss_class0 + loss_class1)*0.5
        else:
            loss = loss_class1

        macro_loss = loss.mean()
        return macro_los
