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

class ConvNeXtBlock(nn.Module):
    """Lightweight modern CNN block using large depthwise kernels for robust receptive fields."""
    def __init__(self, dim):
        super().__init__()
        # Depthwise convolution preserves spatial details separately per channel
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        input_x = x
        x = self.dwconv(x)
        # Permute to channels-last for standard LayerNorm/Linear processing
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return input_x + x


class HierarchicalLiDARCNNMAE(nn.Module):
    def __init__(
        self, 
        in_chans=24,         # Cropped Z-bins from your npz files
        embed_dim=768,       # Final bottleneck dimension for GameFormer
        mask_ratio=0.75,     # Mask out 75% of the spatial layout
        mask_patch_size=32   # Drop large contiguous blocks to force semantic inpainting
    ):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.mask_patch_size = mask_patch_size

        # ----------------------------------------------------------------------
        # 1. ENCODER: Gentle, Progressive Downsampling (748 -> 374 -> 187 -> 94 -> 47 -> 24)
        # ----------------------------------------------------------------------
        
        # Stem: Gentle 2x reduction (748 -> 374)
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        self.enc_stage1 = nn.Sequential(ConvNeXtBlock(64), ConvNeXtBlock(64))

        # Downsample 1: (374 -> 187)
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128)
        )
        self.enc_stage2 = nn.Sequential(ConvNeXtBlock(128), ConvNeXtBlock(128))

        # Downsample 2: Handles odd dimension cleanly (187 -> 94) via kernel=3, pad=1
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256)
        )
        self.enc_stage3 = nn.Sequential(ConvNeXtBlock(256), ConvNeXtBlock(256), ConvNeXtBlock(256))

        # Downsample 3: (94 -> 47)
        self.down3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512)
        )
        self.enc_stage4 = nn.Sequential(ConvNeXtBlock(512), ConvNeXtBlock(512))

        # Downsample 4: Handles odd dimension cleanly (47 -> 24) to hit the exact bottleneck
        self.down4 = nn.Sequential(
            nn.Conv2d(512, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim)
        )
        self.enc_stage5 = nn.Sequential(ConvNeXtBlock(embed_dim), ConvNeXtBlock(embed_dim))

        # Learnable embedding to fill in missing/masked gaps at the bottleneck
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim, 1, 1))
        
        # ----------------------------------------------------------------------
        # 2. DECODER: Generative Progressive Upsampling (U-Net Style)
        # ----------------------------------------------------------------------
        self.up4 = nn.Conv2d(embed_dim, 512, kernel_size=1)
        self.dec_stage4 = nn.Sequential(
            nn.Conv2d(512 + 512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            ConvNeXtBlock(512)
        )

        self.up3 = nn.Conv2d(512, 256, kernel_size=1)
        self.dec_stage3 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            ConvNeXtBlock(256)
        )

        self.up2 = nn.Conv2d(256, 128, kernel_size=1)
        self.dec_stage2 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            ConvNeXtBlock(128)
        )

        self.up1 = nn.Conv2d(128, 64, kernel_size=1)
        self.dec_stage1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            ConvNeXtBlock(64)
        )

        # Final projection back to native 748x748 resolution and 24 Z-bins
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, in_chans, kernel_size=3, padding=1)
        )

        nn.init.normal_(self.mask_token, std=0.02)


    def generate_patch_mask(self, x):
        """Generates a contiguous binary block mask: 1 means masked (hidden), 0 means visible."""
        B, _, H, W = x.shape
        P = self.mask_patch_size
        
        # Calculate grid size for mask blocks
        grid_h, grid_w = H // P, W // P
        
        # Generate random noise and threshold to hit the exact 75% mask ratio
        noise = torch.rand(B, 1, grid_h, grid_w, device=x.device)
        mask = (noise < self.mask_ratio).float()
        
        # Upsample the coarse mask blocks to the full 748x748 physical grid
        full_mask = F.interpolate(mask, size=(H, W), mode='nearest')
        return full_mask


    def forward_encoder(self, x, mask=None):
        """Processes input through gentle downsampling stages and returns multi-scale features."""
        # Apply mask directly in dense space to wipe out hidden features
        if mask is not None:
            x = x * (1.0 - mask)

        # Stem: 748 -> 374
        feat1 = self.enc_stage1(self.stem(x))
        
        # Stage 2: 374 -> 187
        feat2 = self.enc_stage2(self.down1(feat1))
        
        # Stage 3: 187 -> 94
        feat3 = self.enc_stage3(self.down2(feat2))
        
        # Stage 4: 94 -> 47
        feat4 = self.enc_stage4(self.down3(feat3))
        
        # Bottleneck Stage 5: 47 -> 24
        bottleneck = self.enc_stage5(self.down4(feat4))
        
        return bottleneck, [feat1, feat2, feat3, feat4]


    def forward_decoder(self, bottleneck, skips, mask):
        """Reconstructs the full spatial layout using U-Net skip connections and dynamic upsampling."""
        feat1, feat2, feat3, feat4 = skips
        
        # Fill in masked areas at the bottleneck using the learnable mask token
        # Downsample the binary mask to the 24x24 bottleneck shape
        b_mask = F.interpolate(mask, size=bottleneck.shape[2:], mode='nearest')
        bottleneck = bottleneck * (1.0 - b_mask) + self.mask_token * b_mask

        # ----------------------------------------------------------------------
        # Progressive Upsampling with Explicit Size Matching
        # ----------------------------------------------------------------------
        
        # Up 4: 24 -> 47 (Matches Skip 4 cleanly)
        d4 = self.up4(bottleneck)
        d4 = F.interpolate(d4, size=feat4.shape[2:], mode='bilinear', align_corners=False)
        d4 = self.dec_stage4(torch.cat([d4, feat4], dim=1))

        # Up 3: 47 -> 94 (Matches Skip 3 cleanly)
        d3 = self.up3(d4)
        d3 = F.interpolate(d3, size=feat3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec_stage3(torch.cat([d3, feat3], dim=1))

        # Up 2: 94 -> 187 (Matches Skip 2 cleanly)
        d2 = self.up2(d3)
        d2 = F.interpolate(d2, size=feat2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec_stage2(torch.cat([d2, feat2], dim=1))

        # Up 1: 187 -> 374 (Matches Skip 1 cleanly)
        d1 = self.up1(d2)
        d1 = F.interpolate(d1, size=feat1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec_stage1(torch.cat([d1, feat1], dim=1))

        # Final Up: 374 -> 748 native output
        reconstruction = self.final_up(d1)
        return reconstruction


    def forward(self, x):
        """End-to-End Pretraining Pass: Computes masked MSE reconstruction loss."""
        # 1. Generate the block mask
        mask = self.generate_patch_mask(x)

        # 2. Extract multi-scale hierarchy
        bottleneck, skips = self.forward_encoder(x, mask)

        # 3. Reconstruct full layout
        reconstruction = self.forward_decoder(bottleneck, skips, mask)

        # ----------------------------------------------------------------------
        # SMART LOSS: Compute Mean Squared Error strictly on masked regions
        # ----------------------------------------------------------------------
        mse_loss = (reconstruction - x) ** 2
        
        # Zero out errors on unmasked/visible pixels, average strictly across the hidden ones
        masked_mse_loss = (mse_loss * mask).sum() / (mask.sum() + 1e-8)

        return masked_mse_loss


    def forward_features(self, lidar_sequence):
        """
        GAMEFORMER EXTRACTION WRAPPER:
        Accepts raw sequence: [B, 11, 24, 748, 748].
        Folds time, pools the bottleneck, and unfolds back to sequential timeline: [B, 11, 768].
        """
        self.eval() # Lock batchnorm/dropout
        
        B, T, C, H, W = lidar_sequence.shape
        x_folded = lidar_sequence.to(torch.float32).view(B * T, C, H, W)

        with torch.no_grad():
            # Process complete frames through the encoder without any masking
            bottleneck_maps, _ = self.forward_encoder(x_folded, mask=None)
            
            # Bottleneck maps shape: [B * 11, 768, 24, 24]
            # Apply Global Average Pooling to collapse the 24x24 spatial footprint
            pooled = F.adaptive_avg_pool2d(bottleneck_maps, (1, 1)).flatten(start_dim=1)
            
            # Unfold time back to sequential output for cross-attention
            gameformer_embeddings = pooled.view(B, T, self.embed_dim)

        return gameformer_embeddings
    
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