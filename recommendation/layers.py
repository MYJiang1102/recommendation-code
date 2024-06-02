import torch
from torch import nn
from torch.nn.parameter import Parameter
from abc import abstractmethod
from torch.nn import Parameter,init
import pandas as pd
import torch.nn.functional as F
LAYER_IDS = {}

#定义线性层
class Dense(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.2, chnl=8):
        super(Dense, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.act = nn.ReLU()
        self.drop_layer = nn.Dropout(p=self.dropout) # Pytorch drop: ratio to zeroed
        self.fc = nn.Linear(self.input_dim, self.output_dim,bias=False)

    def forward(self, inputs):
        x = self.drop_layer(inputs)
        output = self.fc(x)
        return self.act(output)

class Dense2(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0):
        super(Dense2, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.act = nn.Tanh()
        self.drop_layer = nn.Dropout(p=self.dropout) # Pytorch drop: ratio to zeroed
        self.fc = nn.Linear(self.input_dim, self.output_dim)
    def forward(self, inputs):
        x = self.drop_layer(inputs)
        output = self.fc(x)
        return self.act(output)


class CNN_cross( nn.Module ):

    def __init__( self, dim):
        super( CNN_cross, self ).__init__()
        self.dim = dim
        self.Conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2,1))
        self.Conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2, 3))
        self.Conv5 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2, 5))
        self.Conv7 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2, 7))
        # self.Conv9 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2, 9))
        # self.Conv11 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2, 11))
        # self.Conv13= nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2, 13))

        self.ConV = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(4,1))
    def forward(self, input):
        i,h=input
        h=torch.unsqueeze(h,1)
        i = torch.unsqueeze(i, 1)
        #torch.Size([4096, 1, 8])
        HI = torch.cat([i,h], dim=1)
        HI = torch.unsqueeze(HI, 1)
        HI1  = self.Conv1(HI)
        HI3 = self.Conv3(HI)
        HI5 = self.Conv5(HI)
        HI7 = self.Conv7(HI)


        # HI13 = self.Conv13(HI)


        Z3=torch.zeros(HI.shape[0],1,2)
        Z5 = torch.zeros(HI.shape[0], 1, 4)
        Z7 = torch.zeros(HI.shape[0], 1, 6)


        # Z13 = torch.zeros(HI.shape[0], 1, 12)
        HI1= torch.squeeze(HI1, 1)
        HI3 = torch.squeeze(HI3, 1)
        HI5 = torch.squeeze(HI5, 1)
        HI7 = torch.squeeze(HI7, 1)


        # HI13= torch.squeeze(HI13, 1)
        HI3=torch.cat([HI3,Z3],dim=2)
        HI5 = torch.cat([HI5, Z5], dim=2)
        HI7 = torch.cat([HI7, Z7], dim=2)
        # perm3 = torch.randperm(HI3.size(-1))
        # HI3 = HI3[:, :, perm3]
        # perm5 = torch.randperm(HI5.size(-1))
        # HI5 = HI5[:, :, perm5]
        # perm7 = torch.randperm(HI7.size(-1))
        # HI7 = HI7[:, :, perm7]

        # HI13 = torch.cat([HI13, Z13], dim=2)
        HIcat=torch.cat([HI1,HI3,HI5,HI7],dim=1)

        HIcat=torch.unsqueeze(HIcat, 1)

        HIcat=self.ConV(HIcat)
        HIcat = torch.squeeze(HIcat, 2)
        item_out,head_out, = torch.chunk(HIcat, 2, 1)
        item_out = torch.squeeze(item_out, 1)
        head_out = torch.squeeze(head_out, 1)

        return item_out,head_out

class GCN( nn.Module ):

    def __init__( self, dim,batch_size):
        super( GCN, self ).__init__()
        self.dim = dim
        self.batch_size=batch_size
        self.T_Embedding = nn.Embedding(9366, self.dim)
        self.R_Embedding=nn.Embedding(60,self.dim)
        self.U_Embedding = nn.Embedding(1872, self.dim)
    def forward(self, inputs):
        u,v = inputs
        kg_data = pd.read_table('D:\Pycharm\推荐系统\Test\KGCN-master\MKR.PyTorch-master - music\data\music\\kg_final.txt',
                                sep='	', header=None, names=['H', 'R', 'T'])
        # RC_data = pd.read_table('D:\Pycharm\推荐系统\Test\KGCN-master\MKR.PyTorch-master\data\movie\\ratings_final.txt',
        #                         sep='	', header=None, names=['U', 'V', 'Rat'])

        kg_data.index = kg_data['H']
        # RC_data.index = RC_data['U']
        # RC_data = RC_data.drop('U', axis=1)
        kg_data = kg_data.drop('H', axis=1)

        U_v=self.U_Embedding(u)
        u_list=u.detach().numpy().tolist()
        i = u.shape[0]
        v_list = v.detach().numpy().tolist()
        v_c = torch.zeros(i, self.dim)
        j=0
        for vi in v_list:
            R = kg_data.loc[vi]["R"].tolist()
            if isinstance(R, int):
                R = [R]
            R=torch.LongTensor(R)
            R_v=self.R_Embedding(R)
            T = kg_data.loc[vi]["T"].tolist()
            if isinstance(T,int):
                T=[T]
            T=torch.LongTensor(T)
            T_v=self.T_Embedding(T)

            # V_New = 0
            # p=0
            score=[]
            for k in R_v:

                U_R_score = torch.sum(U_v[j] * k)
                score.append(U_R_score)
                # print(U_R_score)
                # # print(U_R_score)
                # V_new = T_v[p] * U_R_score
                # V_New+=V_new
                # p+=1
            score=torch.Tensor(score)
            score = F.softmax(score, -1)
            score=torch.unsqueeze(score,1)

            V_New=torch.sum(score*T_v,axis=0)


            v_c[j]=V_New

            j+=1
        return v_c

class HCN( nn.Module ):

    def __init__( self, dim,batch_size):
        super( HCN, self ).__init__()
        self.dim = dim
        self.batch_size=batch_size
        self.T_Embedding = nn.Embedding(9366, self.dim)
        self.H_Embedding = nn.Embedding(3846, self.dim)
        self.R_Embedding=nn.Embedding(60,self.dim)
    def forward(self, input):
        h=input
        h=h[0]
        kg_data = pd.read_table('D:\Pycharm\推荐系统\Test\KGCN-master\MKR.PyTorch-master - music\data\music\\kg_final.txt',
                                sep='	', header=None, names=['H', 'R', 'T'])
        kg_data.index = kg_data['H']
        # RC_data.index = RC_data['U']
        # RC_data = RC_data.drop('U', axis=1)
        kg_data = kg_data.drop('H', axis=1)

        i = h.shape[0]
        h_list = h.detach().numpy().tolist()
        v_c = torch.zeros(i, self.dim)
        H_v=self.H_Embedding(h)
        j=0
        for hi in h_list:
            R = kg_data.loc[hi]["R"].tolist()
            if isinstance(R, int):
                R = [R]
            R=torch.LongTensor(R)
            R_v=self.R_Embedding(R)
            T = kg_data.loc[hi]["T"].tolist()
            if isinstance(T, int):
                T = [T]
            T = torch.LongTensor(T)
            T_v = self.T_Embedding(T)

            # V_New = 0
            # p=0
            score=[]
            for k in R_v:
                H_R_score = torch.sum(H_v[j] * k)
                score.append(H_R_score)
                # print(U_R_score)
                # # print(U_R_score)
                # V_new = T_v[p] * U_R_score
                # V_New+=V_new
                # p+=1
            score=torch.Tensor(score)
            score = F.softmax(score, -1)
            score=torch.unsqueeze(score,1)
            Hv=(T_v-R_v)*score
            V_New=torch.sum(Hv)


            v_c[j]=V_New

            j+=1
        return v_c

class UCN( nn.Module ):

    def __init__( self, dim,batch_size):
        super( UCN, self ).__init__()
        self.dim = dim
        self.batch_size=batch_size
        self.U_Embedding = nn.Embedding(1872, self.dim)
        self.V_Embedding = nn.Embedding(3846, self.dim)

    def forward(self, input):
        u,v=input

        rat_data = pd.read_table('D:\Pycharm\推荐系统\Test\KGCN-master\MKR.PyTorch-master - music\data\music\\ratings_final.txt', sep = '	', header = None
                                , names = ['U','V','I'] )
        rat_data.index = rat_data['V']
        # RC_data.index = RC_data['U']
        # RC_data = RC_data.drop('U', axis=1)
        rat_data =  rat_data.drop('V', axis=1)

        i = u.shape[0]
        u_list = u.detach().numpy().tolist()
        v_list = v.detach().numpy().tolist()
        u_c = torch.zeros(i, self.dim)

        j=0
        for vi in v_list:
            U = rat_data.loc[vi]["U"].tolist()
            if isinstance(U, int):
                U = [U]
            U = torch.LongTensor(U)
            U_v = self.U_Embedding(U)
            U_v=torch.sum(U_v,0)
            u_c[j]=torch.unsqueeze(U_v,0)
        return u_c















#定义交叉压缩层
# class CrossCompressUnit( nn.Module ):
#
#     def __init__( self, dim ):
#         super( CrossCompressUnit, self ).__init__()
#         self.dim = dim
#
#         self.weight_vv = init.xavier_uniform_( Parameter( torch.empty( dim, 1 ) ) )
#         self.weight_ev = init.xavier_uniform_( Parameter( torch.empty( dim, 1 ) ) )
#         self.weight_ve = init.xavier_uniform_( Parameter( torch.empty( dim, 1 ) ) )
#         self.weight_ee = init.xavier_uniform_( Parameter( torch.empty( dim, 1 ) ) )
#
#         self.bias_v = init.xavier_uniform_( Parameter( torch.empty( 1, dim ) ) )
#         self.bias_e = init.xavier_uniform_( Parameter( torch.empty( 1, dim ) ) )
#
#     def forward( self, inputs):
#         v,e=inputs
#         # [ batch_size, dim ]
#         # [ batch_size, dim, 1 ]
#         v = v.reshape( -1, self.dim, 1 )
#         # [ batch_size, 1, dim ]
#         e = e.reshape( -1, 1, self.dim )
#         # [ batch_size, dim, dim ]
#         c_matrix = torch.matmul( v, e )
#         # [ batch_size, dim, dim ]
#         c_matrix_transpose = torch.transpose( c_matrix, dim0 = 1, dim1 = 2 )
#         # [ batch_size * dim, dim ]
#         c_matrix = c_matrix.reshape( ( -1, self.dim ) )
#         c_matrix_transpose = c_matrix_transpose.reshape( ( -1, self.dim ) )
#         # [batch_size, dim]
#         v_output = torch.matmul( c_matrix, self.weight_vv ) + torch.matmul( c_matrix_transpose, self.weight_ev )
#         e_output = torch.matmul( c_matrix, self.weight_ve ) + torch.matmul( c_matrix_transpose, self.weight_ee )
#         # [batch_size, dim]
#         v_output = v_output.reshape( -1, self.dim ) + self.bias_v
#         e_output = e_output.reshape( -1, self.dim ) + self.bias_e
#         return v_output, e_output