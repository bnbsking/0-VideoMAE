import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from modeling_finetune import Block, _cfg, PatchEmbed, get_sinusoid_encoding_table
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


__all__ = [
    'pretrain_videomae_base_patch16_224', 
    'pretrain_videomae_large_patch16_224', 
]


class PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, tubelet_size=2,
                 use_learnable_pos_emb=False): # 224, 16, 3, 0, 768, 12, 12, 4, True*, None, 0, 0, 0, nn.LayerNorm, None, 2, False
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed( # B,3,16,224,224 -> B,784,8,14,14 -> B,1568(f),768(c)
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,tubelet_size=tubelet_size) 
        num_patches = self.patch_embed.num_patches # 196

        # TODO: Add the cls token
        if use_learnable_pos_emb: # False
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings 
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim) # shape=(1,1568,768)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule # []
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, mask): # (B,3,16,224,224),(B,1568)
        _, _, T, _, _ = x.shape
        x = self.patch_embed(x) # B,1568(f),768(c)
        
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach() # B,1568(F),768(C)

        B, _, C = x.shape
        # e.g. B=16, x[~mask].shape=(2560,768) # 16*1568*(20/196)=2560 # B*F*unmask_ratio=unmask_feature(UF)
        x_vis = x[~mask].reshape(B, -1, C) # ~mask means visible # x[~mask].shape=(2560,768) # B,UF/B,C

        for blk in self.blocks:
            x_vis = blk(x_vis) # B,160,768

        x_vis = self.norm(x_vis) # B,160,768
        return x_vis

    def forward(self, x, mask): # (B,3,16,224,224),(B,1568)
        x = self.forward_features(x, mask) # B,160,768
        x = self.head(x) # B,160,768
        return x

class PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, patch_size=16, num_classes=768, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, num_patches=196, tubelet_size=2
                 ): # 16, 1536, 384, 12, 12, 4, True, None, 0, 0, 0, nn.LayerNorm, 0, 1568, 2
        super().__init__()
        self.num_classes = num_classes
        assert num_classes == 3 * tubelet_size * patch_size ** 2 # 3*2*16**2=1536
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models # 384
        self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule # []
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values) # 384,12,4,True,None,0,0,0,0,nn.LayerNorm,0
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, return_token_num): # (B,F=1568,C=384), 1408
        for blk in self.blocks: # B,1568,384
            x = blk(x)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:])) # only return the mask tokens predict pixels # B,1408,1536
        else:
            x = self.head(self.norm(x))

        return x

class PretrainVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224, 
                 patch_size=16, 
                 encoder_in_chans=3, 
                 encoder_num_classes=0, 
                 encoder_embed_dim=768,
                 encoder_depth=12,
                 encoder_num_heads=12, 
                 decoder_num_classes=1536, # 1536 # (X) decoder_num_classes=768, 
                 decoder_embed_dim=512, # 384
                 decoder_depth=8,
                 decoder_num_heads=8, # 6
                 mlp_ratio=4., 
                 qkv_bias=False, # True
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 tubelet_size=2,
                 num_classes=0, # avoid the error from create_fn in timm
                 in_chans=0, # avoid the error from create_fn in timm
                 ):
        super().__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size, # 224
            patch_size=patch_size, # 16
            in_chans=encoder_in_chans, # 3
            num_classes=encoder_num_classes, # 0
            embed_dim=encoder_embed_dim, # 768
            depth=encoder_depth, # 12
            num_heads=encoder_num_heads, # 12 
            mlp_ratio=mlp_ratio, # 4
            qkv_bias=qkv_bias, # True
            qk_scale=qk_scale, # None
            drop_rate=drop_rate, # 0
            attn_drop_rate=attn_drop_rate, # 0
            drop_path_rate=drop_path_rate, # 0
            norm_layer=norm_layer, # nn.LayerNorm
            init_values=init_values, # 0
            tubelet_size=tubelet_size, # 2
            use_learnable_pos_emb=use_learnable_pos_emb) # False

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size, # 16
            num_patches=self.encoder.patch_embed.num_patches, # 1568
            num_classes=decoder_num_classes, # 1536
            embed_dim=decoder_embed_dim, # 384
            depth=decoder_depth, # 8
            num_heads=decoder_num_heads, # 12
            mlp_ratio=mlp_ratio, # 4
            qkv_bias=qkv_bias, # True
            qk_scale=qk_scale, # None
            drop_rate=drop_rate, # 0
            attn_drop_rate=attn_drop_rate, # 0
            drop_path_rate=drop_path_rate, # 0
            norm_layer=norm_layer, # nn.LayerNorm
            init_values=init_values, # 0
            tubelet_size=tubelet_size) # 2

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False) # 768,384

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim)) # 1,1,384

        self.pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches, decoder_embed_dim) # (1,1568,384)

        trunc_normal_(self.mask_token, std=.02)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, mask): # (B,3,16,224,224), (B,1568)
        _, _, T, _, _ = x.shape
        x_vis = self.encoder(x, mask) # [B, N_vis, C_e] # (B,160,768)
        x_vis = self.encoder_to_decoder(x_vis) # [B, N_vis, C_d] # (B,160,384)
        B, N, C = x_vis.shape
        # we don't unshuffle the correct visible token order, 
        # but shuffle the pos embedding accorddingly.
        # e.g. [[1], [2], [3]] --.expand(3,4)--> [[1,1,1,1], [2,2,2,2], [3,3,3,3]] # this case (1,1568,384) -> (B,1568,384)
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1) # [B, N, C_d]
        x = self.decoder(x_full, pos_emd_mask.shape[1]) # [B, N_mask, 3 * 16 * 16]

        return x

@register_model
def pretrain_mae_small_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=6,
        encoder_num_classes=0,
        decoder_num_classes=1536, 
        decoder_embed_dim=192, 
        decoder_num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def pretrain_videomae_base_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16, 
        encoder_embed_dim=768, 
        encoder_depth=12, 
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=1536,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
 
@register_model
def pretrain_videomae_large_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16, 
        encoder_embed_dim=1024, 
        encoder_depth=24, 
        encoder_num_heads=16,
        encoder_num_classes=0,
        decoder_num_classes=1536, 
        decoder_embed_dim=512,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
