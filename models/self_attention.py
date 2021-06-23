import torch.nn as nn
import torch.nn.functional as F
import torch, math
import copy

    


class MHAtt(nn.Module):
    def __init__(self, embedding_size = 256, multi_head=1,dropout_ratio = 0.1):
        super(MHAtt, self).__init__()
        
        self.multi_head = multi_head
        self.multi_head_size = int(embedding_size/multi_head)
        
        self.embedding_size = embedding_size
        
        self.linear_v = nn.Linear(embedding_size, embedding_size)
        self.linear_k = nn.Linear(embedding_size, embedding_size)
        self.linear_q = nn.Linear(embedding_size, embedding_size)
        self.linear_merge = nn.Linear(embedding_size, embedding_size)

        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, q, k, v):
        # input shape B x HW x C
        B = q.shape[0]
        v = self.linear_v(v).view(
            B,
            -1,
            self.multi_head,
            self.multi_head_size
        ).transpose(1, 2)
        k = self.linear_k(k).view(
            B,
            -1,
            self.multi_head,
            self.multi_head_size
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            B,
            -1,
            self.multi_head,
            self.multi_head_size
        ).transpose(1, 2)
        
        atted = self.att(v, k, q)
        atted = atted.transpose(1, 2).contiguous().view(
            B,
            -1,
            self.embedding_size
        )

        atted = self.linear_merge(atted)
        
        return atted

    def att(self, value, key, query, mask=None):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        
        att_info = {'att_map':att_map, 'query':query, 'key':key, 'value': value}
        
        att_map = self.dropout(att_map)
        
        return torch.matmul(att_map, value)



def _get_clones(module,num_layers):
        return nn.ModuleList([copy.deepcopy(module) for i in range(num_layers)])
    
class Transformer(nn.Module):
    def __init__(self, dropout_ratio = 0.1, multi_head = 1, embedding_size = 256, pre_normalize=\
                 False, num_self_attention_layers = 1,**kwargs):
        super(Transformer, self).__init__()
        self.layers = _get_clones(SA(dropout_ratio = dropout_ratio,multi_head = multi_head, embedding_size = embedding_size, pre_normalize = pre_normalize), num_layers = num_self_attention_layers)
        
        
    
    
    def forward(self,x, shape = {}, pos = None):
        output = x
        
        for layer in self.layers:
            output = layer(output, shape = shape, pos = pos) 
        return output
        
        
        
        
    
    
class SA(nn.Module):
    def __init__(self, dropout_ratio = 0.1, multi_head = 1, embedding_size = 256, pre_normalize = False):
        super(SA, self).__init__()
        self.mhatt = MHAtt(embedding_size = embedding_size, multi_head = multi_head)
        self.embedding_size = embedding_size
        
        self.dropout = nn.Dropout(dropout_ratio)
        self.dropout1 = nn.Dropout(dropout_ratio)
        self.norm1 = nn.LayerNorm(embedding_size)

        self.dropout2 = nn.Dropout(dropout_ratio)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.linear1 = nn.Linear(embedding_size, embedding_size)
        self.linear2 = nn.Linear(embedding_size, embedding_size)
        self.pre_normalize = pre_normalize
        
 
        
    def forward(self, x, shape = {}, pos = None):
        
        if self.pre_normalize:
            return self.forward_pre(x, shape, pos)
        return self.forward_post(x, shape, pos)
        
    
    def forward_pre(self, x, shape = {}, pos = None) :
        
        v = x.view(shape['B'], -1, self.embedding_size)
        v2 = self.norm1(v)   
        q = k = self.with_pos_embed(v2.view(shape['B'], shape['H'], shape['W'],-1), pos)
        q, k = q.view(shape['B'], -1, self.embedding_size),k.view(shape['B'], -1, self.embedding_size) 
        v = v + self.dropout1(self.mhatt(q, k , v2))
        v2 = self.norm2(v)
        v2 = self.linear2(self.dropout(F.relu(self.linear1(v2))))
        v = v + self.dropout2(v2)
        
        return v
        
        
    def with_pos_embed(self, x, pos = None):
        return x if pos is None else x + pos
        
    
    def forward_post(self, x, shape = {},pos = None):
         
        q = k = self.with_pos_embed(x , pos)
        q, k = q.view(shape['B'], -1, self.embedding_size), k.view(shape['B'], -1, self.embedding_size)
        v = x.view(shape['B'], -1, self.embedding_size)
        
        v_temp = v
        
        atted = self.mhatt(q, k , v)
            
            
        x = self.norm1(v + self.dropout1(atted))
        x2 = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = self.norm2(x + self.dropout2(x2))
        
        
        x = x.view(shape['B'],shape['H'], shape['W'],-1)

        return x

