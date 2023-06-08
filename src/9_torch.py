import torch
import torch.nn as nn


def replace_module_(module: nn.Module, to_replace, replace_with: nn.Module):
    """
    replaces all modules of the class to_replace with the object replace_with.

    :param module:
    :param to_replace: a class inheriting from torch.nn.Module that has to be replaced

    :param replace_with: module to use instead

    :return:
    """
    for name, child in module.named_children():
        if isinstance(child, to_replace):
            setattr(module, name, replace_with)
        else:
            replace_module_(child, to_replace, replace_with)


class Net(nn.Module):
    def __init__(self, n_modules: int):
        super(Net, self).__init__()
        self.module_list = nn.ModuleList()
        for _ in range(n_modules):
            self.module_list.extend([nn.Linear(in_features=2, out_features=2), nn.ReLU()])

    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        return x


def greater_zero(input):
    if input is not None:
        return (input > 0).int() * input
    return None


def rectify_hook(module: nn.Module, grad_inputs: torch.Tensor, save_list):
    new_grad_inputs = tuple([greater_zero(gi) for gi in grad_inputs])
    print("old grad inputs:", grad_inputs)
    print("new grad inputs:", new_grad_inputs)
    return new_grad_inputs


net = Net(n_modules=2)
replace_module_(module=net, to_replace=nn.ReLU, replace_with=nn.Sigmoid())
save_list = []

for module in net.modules():
    module.register_backward_hook(lambda module, grad_input, save_list: rectify_hook(module, grad_input, save_list=save_list))


x = torch.tensor([[-4., 5.],
                  [6., -7.]])
pred = net(x)[1][1]
pred.backward()
print("\n\n")


print("grads:")
for name, parameter in net.named_parameters():
    print(name, parameter.grad)
    print(torch.norm(parameter.grad, p=2))

torch.nn.utils.clip_grad_norm_(net.parameters(), 1)

print("\n\ngrads after clipping:")
for name, parameter in net.named_parameters():
    print(name, parameter.grad)