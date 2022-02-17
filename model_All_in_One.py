import torch
import torch.nn as nn
import torch.optim

import FrEIA.framework as Ff
import FrEIA.modules as Fm

ndim_total = 3



class cINN(nn.Module):
    '''cINN for class-conditional MNISt generation'''
    def __init__(self, lr):
        super().__init__()

        self.cinn = self.build_inn()

        self.trainable_parameters = [p for p in self.cinn.parameters() if p.requires_grad]
        for p in self.trainable_parameters:
            p.data = 0.01 * torch.randn_like(p)

        self.optimizer = torch.optim.Adam(self.trainable_parameters, lr=lr, weight_decay=1e-5)

    def build_inn(self):
        print('Entered build')
        def subnet(ch_in, ch_out):
            print('ch_in and ch_out are', str(ch_in) , str(ch_out))  # 402, 784
            return nn.Sequential(nn.Linear(ch_in, 100),
                                 nn.ReLU(),
                                 nn.Linear(100, ch_out))

        cond = Ff.ConditionNode(34)
        input_dims = (3,)  # was 35
#        nodes = [Ff.InputNode(1, 28, 28)]
        
        cond_dims = (34,)
#        cinn = Ff.SequenceINN(3)
        
        
        nodes = [Ff.InputNode(3)]
        nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}))
        
#        nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}))

        for k in range(8):

            nodes.append(Ff.Node(nodes[-1], Fm.AllInOneBlock,
                                 {'subnet_constructor':subnet},
                                 conditions=cond))

        return Ff.ReversibleGraphNet(nodes + [cond, Ff.OutputNode(nodes[-1])], verbose=True)
    

    def forward(self, x, l):

#        --------------------------
        z,jac = self.cinn(x , c=l, jac= True)
        return z, jac

    def reverse_sample(self, z, l):
        return self.cinn(z, c=l, rev=True)

