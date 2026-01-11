class HierarchicalAttention(nn.Module):
    def __init__(self, channels, hierarchy_levels=[2, 4], reduction_ratio=16, channel_attention_type='se', eca_kernel_size=3, allocation_mode='softmax'):
        super(HierarchicalAttention, self).__init__()
        self.channels = channels
        self.hierarchy_levels = hierarchy_levels
        self.num_levels = len(hierarchy_levels)
        self.allocation_mode = allocation_mode
        
        # 为每个层次创建可学习的权重
        self.level_weights = nn.Parameter(torch.ones(self.num_levels))

        # 通道注意力
        if channel_attention_type == 'se':
            mid_channels = max(1, channels // reduction_ratio)
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, mid_channels, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, channels, 1, bias=False),
                nn.Sigmoid()
            )
        elif channel_attention_type == 'srm':
            self.channel_attention = SRMAttention(channels)
        elif channel_attention_type == 'eca':
            self.channel_attention = ECAAttention(channels, k_size=eca_kernel_size)
        else:
            mid_channels = max(1, channels // reduction_ratio)
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, mid_channels, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, channels, 1, bias=False),
                nn.Sigmoid()
            )
        
        # Token QKV
        self.token_dim = max(4, min(32, channels // 4))
        
        # Coarse Level SA Projections (Input: 1 -> Dim)
        self.coarse_q = nn.Linear(1, self.token_dim)
        self.coarse_k = nn.Linear(1, self.token_dim)
        self.coarse_v = nn.Linear(1, self.token_dim)
        
        # Cross Level Projections
        # Q comes from previous level's updated token (dim -> dim)
        self.block_q = nn.Linear(self.token_dim, self.token_dim)
        # K, V come from current level's raw token (1 -> dim)
        self.block_k = nn.Linear(1, self.token_dim)
        self.block_v = nn.Linear(1, self.token_dim)
        
        # Output Projection (dim -> 1)
        self.block_out = nn.Linear(self.token_dim, 1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 计算需要填充的尺寸（确保能被所有层次整除）
        max_divisor = math.lcm(*self.hierarchy_levels)
        padded_H = self._get_padded_size(H, max_divisor)
        padded_W = self._get_padded_size(W, max_divisor)
        
        # 如果需要，进行填充
        if H != padded_H or W != padded_W:
            x_padded = F.pad(x, (0, padded_W - W, 0, padded_H - H), mode='replicate')
        else:
            x_padded = x
        
        # 1. 提取所有层次的 Token (Scalar per block)
        raw_tokens = []
        for i, level in enumerate(self.hierarchy_levels):
            block_size = padded_H // level
            # [B, C, level, block, level, block]
            x_reshaped = x_padded.view(B, C, level, block_size, level, block_size)
            x_perm = x_reshaped.permute(0, 1, 2, 4, 3, 5) # [B, C, L, L, block, block]
            # Pool to [B, 1, L, L] -> [B, L*L, 1]
            t_scalar = x_perm.mean(dim=(1, 4, 5)) # Mean over C and spatial block
            raw_tokens.append(t_scalar.view(B, level * level, 1)) # [B, L*L, 1]
        
        level_attentions = []
        last_updated_token = None # [B, P*P, dim]
        
        for i, level in enumerate(self.hierarchy_levels):
            block_size = padded_H // level
            t = raw_tokens[i] # [B, L*L, 1]
            
            if i == 0:
                # Coarse Level Self-Attention
                q = self.coarse_q(t) # [B, L*L, dim]
                k = self.coarse_k(t)
                v = self.coarse_v(t)
                
                # Standard SA: Softmax(QK^T)V
                scores = torch.bmm(q, k.transpose(1, 2)) / (self.token_dim ** 0.5)
                attn = torch.softmax(scores, dim=-1)
                t_updated = torch.bmm(attn, v) # [B, L*L, dim]
                
                last_updated_token = t_updated
                
                # Generate weights
                w_val = self.block_out(t_updated) # [B, L*L, 1]
                w = torch.sigmoid(w_val).view(B, level, level)
                
            else:
                # Cross Level Attention: Parent Q -> Child K
                # Parent tokens from last level
                parent_level = self.hierarchy_levels[i-1]
                parent_tokens = last_updated_token # [B, P*P, dim]
                
                # Map Parent Q to Children
                parent_block = padded_H // parent_level
                
                # Calculate parent indices for each child
                ratio_i = (torch.arange(level, device=x.device) * block_size) // parent_block
                ratio_j = (torch.arange(level, device=x.device) * block_size) // parent_block
                pi = ratio_i.clamp(max=parent_level - 1)
                pj = ratio_j.clamp(max=parent_level - 1)
                grid_pi, grid_pj = torch.meshgrid(pi, pj, indexing='ij')
                parent_idx = (grid_pi * parent_level + grid_pj).view(-1) # [L*L]
                
                # Q from Parent
                q_parent_feat = self.block_q(parent_tokens) # [B, P*P, dim]
                
                # Gather Q for each child
                # parent_idx: [L*L] -> expand to [B, L*L, dim]
                parent_idx_exp = parent_idx.unsqueeze(0).unsqueeze(-1).expand(B, level*level, self.token_dim)
                q_mapped = torch.gather(q_parent_feat, 1, parent_idx_exp) # [B, L*L, dim]
                
                # K, V from Child (Current Level)
                k = self.block_k(t) # [B, L*L, dim]
                v = self.block_v(t) # [B, L*L, dim]
                
                # Score: (Q * K).sum
                scores = (q_mapped * k).sum(dim=-1, keepdim=True) / (self.token_dim ** 0.5) # [B, L*L, 1]
                
                if self.allocation_mode == 'softmax':
                    # Grouped Softmax over siblings
                    # Add exp_scores to parent bin
                    # parent_idx: [L*L] -> [B, L*L, 1]
                    p_idx_b = parent_idx.unsqueeze(0).unsqueeze(-1).expand(B, level*level, 1)
                    max_scores = torch.full(
                        (B, parent_level * parent_level, 1),
                        -torch.inf,
                        device=x.device,
                        dtype=scores.dtype,
                    )
                    max_scores.scatter_reduce_(1, p_idx_b, scores, reduce="amax", include_self=True)
                    max_scores_mapped = torch.gather(max_scores, 1, p_idx_b)
                    scores_shifted = scores - max_scores_mapped
                    exp_scores = torch.exp(scores_shifted)
                    denom = torch.zeros(B, parent_level * parent_level, 1, device=x.device, dtype=exp_scores.dtype)
                    denom.scatter_add_(1, p_idx_b, exp_scores)
                    
                    # Gather denom back
                    denom_mapped = torch.gather(denom, 1, p_idx_b)
                    attn = exp_scores / (denom_mapped + 1e-6)
                else:
                    # Sigmoid Independent
                    attn = torch.sigmoid(scores)
                
                # Update Child Token
                t_updated = attn * v # [B, L*L, dim]
                last_updated_token = t_updated
                
                # Generate weights
                w_val = self.block_out(t_updated)
                w = torch.sigmoid(w_val).view(B, level, level)

            # Expand weights to feature map size
            weights = w.unsqueeze(1).unsqueeze(-1).unsqueeze(-1) # [B, 1, L, L, 1, 1]
            # Need to broadcast to C
            weights = weights.repeat(1, C, 1, 1, 1, 1)
            attn_map = self._expand_weights(weights, block_size, level)
            level_attentions.append(attn_map)
        
        # 层次权重融合
        level_weights = F.softmax(self.level_weights, dim=0)
        
        # 加权求和所有层次的注意力
        base_hierarchical_attn = torch.zeros_like(x_padded)
        for i, attn in enumerate(level_attentions):
            base_hierarchical_attn += level_weights[i] * attn
        
        # 通道注意力调整
        channel_weights = self.channel_attention(x_padded)
        final_attention = base_hierarchical_attn * channel_weights
        
        # 应用注意力并裁剪回原始尺寸
        output_padded = x_padded * final_attention
        output = output_padded[:, :, :H, :W]
        
        return output
    
    def _compute_block_weights(self, x, block_size, grid_size, attn_module):
        B, C, H, W = x.shape
        x_reshaped = x.view(B, C, grid_size, block_size, grid_size, block_size)
        x_permuted = x_reshaped.permute(0, 1, 2, 4, 3, 5)
        x_spatial_as_channels = x_permuted.contiguous().view(B * C, grid_size * grid_size, block_size, block_size)
        weights = attn_module(x_spatial_as_channels)
        weights = weights.view(B, C, grid_size, grid_size, 1, 1)
        return weights
    
    def _compute_block_channel_weights(self, x, block_size, grid_size):
        B, C, H, W = x.shape
        x_reshaped = x.view(B, C, grid_size, block_size, grid_size, block_size)
        x_permuted = x_reshaped.permute(0, 1, 2, 4, 3, 5)
        x_blocks = x_permuted.contiguous().view(B * grid_size * grid_size, C, block_size, block_size)
        cw = self.channel_attention(x_blocks)
        if grid_size == min(self.hierarchy_levels):
            t = cw.view(B * grid_size * grid_size, C, 1)
            q = self.coarse_sa_q(t)
            k = self.coarse_sa_k(t)
            v = self.coarse_sa_v(t)
            attn = torch.matmul(q, k.transpose(1, 2)) / (self.coarse_sa_dim ** 0.5)
            attn = F.softmax(attn, dim=-1)
            refined = torch.matmul(attn, v)
            refined = self.coarse_sa_out(refined)
            cw = cw * refined.view(B * grid_size * grid_size, C, 1, 1)
        cw = cw.view(B, grid_size, grid_size, C, 1, 1).permute(0, 3, 1, 2, 4, 5)
        return cw
    
    def _expand_weights(self, weights, block_size, grid_size):
        B, C = weights.shape[:2]
        wexp = weights.expand(-1, -1, -1, -1, block_size, block_size)
        wreshaped = wexp.permute(0, 1, 2, 4, 3, 5)
        H = grid_size * block_size
        W = grid_size * block_size
        return wreshaped.contiguous().view(B, C, H, W)
    
    def _compute_block_attention(self, x, block_size, grid_size, attn_module):
        B, C, H, W = x.shape
        
        # 1. 切分并重排：将空间块转换为通道维度
        # x: [B, C, H, W] -> [B, C, grid, block, grid, block]
        x_reshaped = x.view(B, C, grid_size, block_size, grid_size, block_size)
        
        # Permute to: [B, C, grid, grid, block, block]
        x_permuted = x_reshaped.permute(0, 1, 2, 4, 3, 5)
        
        # Reshape to treat (grid*grid) as channels: 
        # 我们希望对每个原始通道C内部，计算grid*grid个小块之间的注意力
        # 所以我们将B和C合并作为新的Batch维度，将grid*grid作为新的Channel维度
        # Target: [B*C, grid*grid, block, block]
        x_spatial_as_channels = x_permuted.contiguous().view(B * C, grid_size * grid_size, block_size, block_size)
        
        # 2. 计算注意力权重
        # weights: [B*C, grid*grid, 1, 1]
        weights = attn_module(x_spatial_as_channels)
        
        # 3. 恢复权重形状并广播
        # weights: [B, C, grid, grid, 1, 1]
        weights = weights.view(B, C, grid_size, grid_size, 1, 1)
        
        # 4. 将权重应用回对应的空间位置
        # 为了应用权重，我们需要将其扩展回原始的空间结构
        # x_permuted: [B, C, grid, grid, block, block]
        # weights expanded: [B, C, grid, grid, 1, 1] -> [B, C, grid, grid, block, block]
        weighted_x_permuted = x_permuted * weights
        
        # 5. 还原回原始图像形状
        # [B, C, grid, grid, block, block] -> [B, C, grid, block, grid, block]
        weighted_x_reshaped = weighted_x_permuted.permute(0, 1, 2, 4, 3, 5)
        
        # [B, C, grid, block, grid, block] -> [B, C, H, W]
        attention_map = weighted_x_reshaped.contiguous().view(B, C, H, W)
        
        # 注意：这里返回的是加权后的特征图（Attention Map），而不是单纯的权重值
        # 因为我们已经乘过了。如果只想返回权重图（Mask），则需要修改上述步骤只处理weights。
        # 根据之前的修改意图，这里应该返回权重图（Mask）。
        
        # 修正：只生成权重图
        # weights: [B, C, grid, grid, 1, 1]
        # 我们需要构造一个全图大小的权重图
        weights_expanded = weights.expand(-1, -1, -1, -1, block_size, block_size)
        # [B, C, grid, grid, block, block] -> [B, C, grid, block, grid, block]
        weights_reshaped = weights_expanded.permute(0, 1, 2, 4, 3, 5)
        # [B, C, H, W]
        final_weight_map = weights_reshaped.contiguous().view(B, C, H, W)
        
        return final_weight_map
    
    def _get_padded_size(self, size, divisor):
        if size % divisor == 0:
            return size
        else:
            return (size // divisor + 1) * divisor
    