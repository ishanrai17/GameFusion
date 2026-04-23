import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, max_len=100):
        super(PositionalEncoding, self).__init__()
        d_model = 256
        dropout = 0.1
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0, 2)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x + self.pe
        
        return self.dropout(x)
    

class AgentEncoder(nn.Module):
    def __init__(self):
        super(AgentEncoder, self).__init__()
        self.motion = nn.LSTM(8, 256, 2, batch_first=True)
        self.type_emb = nn.Embedding(4, 256, padding_idx=0)

    def forward(self, inputs):
        ## First 8 features as trajectory, last feature as type
        traj, _ = self.motion(inputs[:, :, :8])
        output = traj[:, -1]
        type = self.type_emb(inputs[:, -1, 8].int())
        ## additive injection
        output = output + type

        return output
    

class LaneEncoder(nn.Module):
    def __init__(self):
        super(LaneEncoder, self).__init__()
        # encdoer layer
        self.self_line = nn.Linear(3, 128)
        self.left_line = nn.Linear(3, 128)
        self.right_line = nn.Linear(3, 128)
        self.speed_limit = nn.Linear(1, 64)
        self.self_type = nn.Embedding(4, 64, padding_idx=0)
        self.left_type = nn.Embedding(11, 64, padding_idx=0)
        self.right_type = nn.Embedding(11, 64, padding_idx=0)
        self.traffic_light_type = nn.Embedding(9, 64, padding_idx=0)
        self.interpolating = nn.Embedding(2, 64)
        self.stop_sign = nn.Embedding(2, 64)

        # hidden layers
        self.pointnet = nn.Sequential(nn.Linear(512, 384), nn.ReLU(), nn.Linear(384, 256))
        self.position_encode = PositionalEncoding(max_len=100)

    def forward(self, inputs):
        # embedding
        self_line = self.self_line(inputs[..., :3])
        left_line = self.left_line(inputs[..., 3:6])
        right_line = self.right_line(inputs[...,  6:9])
        speed_limit = self.speed_limit(inputs[..., 9].unsqueeze(-1))
        self_type = self.self_type(inputs[..., 10].int())
        left_type = self.left_type(inputs[..., 11].int())
        right_type = self.right_type(inputs[..., 12].int()) 
        traffic_light = self.traffic_light_type(inputs[..., 13].int())
        interpolating = self.interpolating(inputs[..., 14].int()) 
        stop_sign = self.stop_sign(inputs[..., 15].int())

        lane_attr = self_type + left_type + right_type + traffic_light + interpolating + stop_sign
        lane_embedding = torch.cat([self_line, left_line, right_line, speed_limit, lane_attr], dim=-1)
    
        # process
        output = self.position_encode(self.pointnet(lane_embedding))

        return output
    

class CrosswalkEncoder(nn.Module):
    def __init__(self):
        super(CrosswalkEncoder, self).__init__()
        self.point_net = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 256))
    
    def forward(self, inputs):
        output = self.point_net(inputs)

        return output
    

class FutureEncoder(nn.Module):
    def __init__(self):
        super(FutureEncoder, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 256))
        self.type_emb = nn.Embedding(4, 256, padding_idx=0)

    def state_process(self, trajs, current_states):
        M = trajs.shape[2]
        current_states = current_states.unsqueeze(2).expand(-1, -1, M, -1)
        xy = torch.cat([current_states[:, :, :, None, :2], trajs], dim=-2)
        dxy = torch.diff(xy, dim=-2)
        v = dxy / 0.1
        theta = torch.atan2(dxy[..., 1], dxy[..., 0].clamp(min=1e-3)).unsqueeze(-1)
        T = trajs.shape[3]
        size = current_states[:, :, :, None, 5:8].expand(-1, -1, -1, T, -1)
        trajs = torch.cat([trajs, theta, v, size], dim=-1) # (x, y, heading, vx, vy, w, l, h)

        return trajs

    def forward(self, trajs, current_states):
        trajs = self.state_process(trajs, current_states)
        trajs = self.mlp(trajs.detach())
        type = self.type_emb(current_states[:, :, None, 8].int())
        output = torch.max(trajs, dim=-2).values
        output = output + type

        return output


class GMMPredictor(nn.Module):
    def __init__(self, future_len):
        super(GMMPredictor, self).__init__()
        self._future_len = future_len
        self.gaussian = nn.Sequential(nn.Linear(256, 512), nn.ELU(), nn.Dropout(0.1), nn.Linear(512, self._future_len*4))
        self.score = nn.Sequential(nn.Linear(256, 64), nn.ELU(), nn.Dropout(0.1), nn.Linear(64, 1))
    
    def forward(self, input):
        B, M, _ = input.shape
        res = self.gaussian(input).view(B, M, self._future_len, 4) # mu_x, mu_y, log_sig_x, log_sig_y
        score = self.score(input).squeeze(-1)

        return res, score


class SelfTransformer(nn.Module):
    def __init__(self):
        super(SelfTransformer, self).__init__()
        heads, dim, dropout = 8, 256, 0.1
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim*4, dim), nn.Dropout(dropout))

    def forward(self, inputs, mask=None):
        attention_output, _ = self.self_attention(inputs, inputs, inputs, key_padding_mask=mask)
        attention_output = self.norm_1(attention_output + inputs)
        output = self.norm_2(self.ffn(attention_output) + attention_output)

        return output


class CrossTransformer(nn.Module):
    def __init__(self):
        super(CrossTransformer, self).__init__()
        heads, dim, dropout = 8, 256, 0.1
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim*4, dim), nn.Dropout(dropout))

    def forward(self, query, key, value, mask=None):
        attention_output, _ = self.cross_attention(query, key, value, key_padding_mask=mask)
        attention_output = self.norm_1(attention_output)
        output = self.norm_2(self.ffn(attention_output) + attention_output)

        return output


class InitialDecoder(nn.Module):
    def __init__(self, modalities, neighbors, future_len):
        super(InitialDecoder, self).__init__()
        dim = 256
        self._modalities = modalities
        self.multi_modal_query_embedding = nn.Embedding(modalities, dim)
        self.agent_query_embedding = nn.Embedding(neighbors+1, dim)
        self.query_encoder = CrossTransformer()
        self.predictor = GMMPredictor(future_len)
        self.register_buffer('modal', torch.arange(modalities).long())
        self.register_buffer('agent', torch.arange(neighbors+1).long())

    def forward(self, id, current_state, encoding, mask):
        # get query
        multi_modal_query = self.multi_modal_query_embedding(self.modal)
        agent_query = self.agent_query_embedding(self.agent[id])
        multi_modal_agent_query = multi_modal_query + agent_query[None, :]
        query = encoding[:, None, id] + multi_modal_agent_query

        # decode trajectories
        query_content = self.query_encoder(query, encoding, encoding, mask)
        predictions, scores = self.predictor(query_content)

        # post process
        predictions[..., :2] += current_state[:, None, None, :2]

        return query_content, predictions, scores


class InteractionDecoder(nn.Module):
    def __init__(self, future_encoder, future_len):
        super(InteractionDecoder, self).__init__()
        self.interaction_encoder = SelfTransformer()
        self.query_encoder = CrossTransformer()
        self.future_encoder = future_encoder
        self.decoder = GMMPredictor(future_len)

    def forward(self, id, current_states, actors, scores, last_content, encoding, mask):
        B, N, M, T, _ = actors.shape
        
        # encoding the trajectories from the last level 
        multi_futures = self.future_encoder(actors[..., :2], current_states)
        futures = (multi_futures * scores.softmax(-1).unsqueeze(-1)).mean(dim=2) 

        # encoding the interaction using self-attention transformer   
        interaction = self.interaction_encoder(futures, mask[:, :N])

        # append the interaction encoding to the context encoding
        encoding = torch.cat([interaction, encoding], dim=1)
        mask = torch.cat([mask[:, :N], mask], dim=1).clone()
        mask[:, id] = True # mask the agent future itself from last level

        # decoding the trajectories from the current level
        query = last_content + multi_futures[:, id]
        query_content = self.query_encoder(query, encoding, encoding, mask)
        trajectories, scores = self.decoder(query_content)

        # post process
        trajectories[..., :2] += current_states[:, id, None, None, :2]

        return query_content, trajectories, scores
    
    
class LiDAREncoder1(nn.Module):
    def __init__(self):
        super(LiDAREncoder1, self).__init__()
        self.conv1 = nn.Conv3d(12, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=0)
        
        self.fnn_block = nn.Linear(256, 256)

    def forward(self, inputs):
        x = nn.ReLU()(nn.MaxPool3d(kernel_size=(1,2,2))(self.conv1(inputs)))
        x = nn.ReLU()(nn.MaxPool3d(kernel_size=(1,2,2))(self.conv2(x)))
        x = nn.ReLU()(nn.MaxPool3d(kernel_size=(1,2,2))(self.conv3(x)))
        x = x.flatten(2).transpose(1, 2)
        x = self.fnn_block(x)
        return x

class LiDAREncoder2(nn.Module):
    def __init__(self):
        super(LiDAREncoder2, self).__init__()
        self.conv1 = nn.Conv3d(12, 64, kernel_size=3, stride=(2, 4, 4), padding=1)   
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=(2, 4, 4), padding=1)  
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, stride=(3, 4, 4), padding=0)
        self.dropout =nn.Dropout(0.2)
        self.fnn_block = nn.Linear(256, 256)

    def forward(self, inputs):
        x = nn.ReLU()(self.conv1(inputs))
        x = nn.ReLU()(self.conv2(x))
        x = nn.ReLU()(self.conv3(x))
        x = self.dropout(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.fnn_block(x)
        return x

    
class LiDAREncoder3(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        for param in self.cnn.parameters():
            param.requires_grad = False
        for param in self.cnn.layer4.parameters():
            param.requires_grad = True
        self.cnn.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn.fc = nn.Identity()
        self.lstm = nn.LSTM(512, 256, batch_first=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # → (B, 11, 12, 300, 300) = (B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)     
        B, T, C, H, W = x.shape
        # (B*11, 12, 300, 300)
        x = x.reshape(B * T, C, H, W)   
        # (B*11, 512) 
        feats = self.cnn(x)    
         # (B, 11, 512)           
        feats = feats.view(B, T, -1) 
        # (B, 11, 256)
        feats = self.dropout(feats)    
        out, _ = self.lstm(feats)          
        return out

class LiDARVAE(nn.Module):
    def __init__(self, latent_dim=256):
        super(LiDARVAE, self).__init__()
        # Input: [B, 12, 11, 300, 300]
        self.enc_conv1 = nn.Conv3d(12, 32, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1))
        self.enc_bn1 = nn.BatchNorm3d(32) # -> [B, 32, 11, 150, 150]
        
        self.enc_conv2 = nn.Conv3d(32, 64, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1))
        self.enc_bn2 = nn.BatchNorm3d(64) # -> [B, 64, 11, 75, 75]
        
        self.enc_conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 3, 3), padding=(1, 0, 0))
        self.enc_bn3 = nn.BatchNorm3d(128) # -> [B, 128, 11, 25, 25]
        
        self.enc_conv4 = nn.Conv3d(128, 256, kernel_size=(3, 5, 5), stride=(1, 5, 5), padding=(1, 0, 0))
        self.enc_bn4 = nn.BatchNorm3d(256) # -> [B, 256, 11, 5, 5]

        self.spatial_flat_size = 256 * 5 * 5 # 6400
        
        # We process each timestep's spatial features independently into the latent space
        self.fc_mu = nn.Linear(self.spatial_flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.spatial_flat_size, latent_dim)
        
        self.fc_decode = nn.Linear(latent_dim, self.spatial_flat_size)

        # Input: [B, 256, 11, 5, 5]
        self.dec_conv4 = nn.ConvTranspose3d(256, 128, kernel_size=(3, 5, 5), stride=(1, 5, 5), padding=(1, 0, 0))
        self.dec_bn4 = nn.BatchNorm3d(128) # -> [B, 128, 11, 25, 25]
        
        self.dec_conv3 = nn.ConvTranspose3d(128, 64, kernel_size=(3, 3, 3), stride=(1, 3, 3), padding=(1, 0, 0))
        self.dec_bn3 = nn.BatchNorm3d(64) # -> [B, 64, 11, 75, 75]
        
        self.dec_conv2 = nn.ConvTranspose3d(64, 32, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1))
        self.dec_bn2 = nn.BatchNorm3d(32) # -> [B, 32, 11, 150, 150]
        
        self.dec_conv1 = nn.ConvTranspose3d(32, 12, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1))
        # -> [B, 12, 11, 300, 300]

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid() # Use sigmoid to output probabilities for BCE Loss
    def vae_loss_function(self, reconstructed_x, original_x, mu, logvar):
        # We use reduction='sum' so it scales properly with the massive 300x300 grid
        BCE = nn.functional.binary_cross_entropy(reconstructed_x, original_x, reduction='sum')
        
        # KL Divergence forces the latent space to be a smooth, continuous distribution
        # Formula: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def encode(self, x):
        x = self.relu(self.enc_bn1(self.enc_conv1(x)))
        x = self.relu(self.enc_bn2(self.enc_conv2(x)))
        x = self.relu(self.enc_bn3(self.enc_conv3(x)))
        x = self.relu(self.enc_bn4(self.enc_conv4(x)))
        
        B, C, T, H, W = x.shape
        # Permute to [B, T, C, H, W] then flatten spatial dims to [B, T, 6400]
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B, T, -1)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        B, T, _ = z.shape
        
        x = self.relu(self.fc_decode(z))
        # Unflatten back to [B, T, 256, 5, 5] and permute back to [B, 256, T, 5, 5]
        x = x.view(B, T, 256, 5, 5).permute(0, 2, 1, 3, 4).contiguous()
        
        x = self.relu(self.dec_bn4(self.dec_conv4(x)))
        x = self.relu(self.dec_bn3(self.dec_conv3(x)))
        x = self.relu(self.dec_bn2(self.dec_conv2(x)))
        
        # No ReLU or BN on the final layer, just Sigmoid for reconstruction
        x = self.sigmoid(self.dec_conv1(x)) 
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

    def get_embeddings(self, x):
        """
        Call this function when you want to bypass the decoder 
        and generate the deterministic static tensors for GameFormer.
        Returns tensor of shape [B, 11, 256]
        """
        with torch.no_grad():
            mu, _ = self.encode(x)
        return mu
    
# class LiDAREncoder3(nn.Module):
#     def __init__(self):
#         super(LiDAREncoder3, self).__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv2d(12, 32, 3, stride=2, padding=1), nn.ReLU(),
#             nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
#             nn.AdaptiveAvgPool2d(1)
#         )
#         self.proj = nn.Linear(64, 256)

#     def forward(self, inputs):
#         feat = self.cnn(inputs).flatten(1)
#         return self.proj(feat)