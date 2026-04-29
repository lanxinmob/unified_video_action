# Architecture Diagram: MAR + VAE + Diffusion

```mermaid
flowchart TD
  subgraph Inputs
    A_img["Images (B,T,C,H,W)"]
    A_act["History actions (B,L,act_dim)"]
    A_text["Text (CLIP) / language_goal"]
  end

  subgraph VAE
    A_img -->|AutoencoderKL.encode() sample -> z (B,T,Cz,Hz,Wz)| VAEDecode[VAE: patchify]
    VAEDecode -->|patchify -> tokens (B,T,S,token_embed_dim)| IMG_TOK
  end

  subgraph ActionProcessing
    A_act -->|normalize -> history_nactions (B,L,act_dim)| ACT_NORM
    ACT_NORM -->|history_action_proj_cond (Linear) -> (B,L,embed)| ACT_PROJ
    ACT_PROJ -->|repeat_interleave(M = buffer_size_action) -> (B, L*M,embed)| ACT_TOK
  end

  subgraph TextProcessing
    A_text -->|CLIP -> vec (512)| TEXT_CLIP
    TEXT_CLIP -->|text_proj_cond -> (B,embed)| TEXT_PROJ
    TEXT_PROJ -->|repeat(buffer_size_text) -> (B,buffer_size_text,embed)| TEXT_TOK
  end

  IMG_TOK["Image tokens (B,T,S,token_embed_dim)"]
  ACT_TOK["Action tokens (B, T*n_frames*M, encoder_embed_dim)"]
  TEXT_TOK

  %% Concatenate tokens
  TEXT_TOK -->|concat along seq dim| CONCAT
  IMG_TOK --> CONCAT
  ACT_TOK --> CONCAT
  CONCAT -->|proj_cond_x_layer (Linear) -> add pos emb| PROJ

  subgraph MAR_Transformer
    PROJ --> EncoderBlocks["Encoder Blocks (timm Block x D)"]
    EncoderBlocks --> DecoderBlocks["Decoder Blocks (x D)"]
    DecoderBlocks --> z_out["z (B, T*S, decoder_embed_dim)"]
  end

  %% MaskGIT style sampling loop
  z_out -->|conditional -> Diffusion (video) / DiffAct (action)| Diff
  Diff -->|sample() predicts token latents -> fill masked tokens| FillTokens
  FillTokens -->|update mask (mask_by_order) and iterate num_iter times| MAR_Iter["Mask-iter num = num_iter (autoregressive)"]
  MAR_Iter --> EncoderBlocks

  %% After sampling
  FillTokens -->|unpatchify -> images| VAE_Decode["VAE.decode -> reconstructed images"]
  FillTokens -->|action decoder (conv_fc + DiffAct sample) -> actions (B,n_steps,act_dim)| ACTION_OUT

  %% Files & key ops
  classDef code fill:#f8f8f8,stroke:#333,stroke-width:1px;
  VAEDecode:::code
  IMG_TOK:::code
  ACT_PROJ:::code
  PROJ:::code
  EncoderBlocks:::code
  Diff:::code

  click VAEDecode "unified_video_action/vae/vaekl.py" "VAE: AutoencoderKL.encode/decode"
  click IMG_TOK "unified_video_action/utils/data_utils.py" "extract_latent_autoregressive / patchify usage"
  click EncoderBlocks "unified_video_action/model/autoregressive/mar_con_unified.py" "forward_mae_encoder / forward_mae_decoder / sample_tokens"
  click Diff "unified_video_action/model/autoregressive/diffusion_loss.py" "DiffLoss / SimpleMLPAdaLN (cond_embed + AdaLN)"
  click ACTION_OUT "unified_video_action/model/autoregressive/diffusion_action_loss.py" "DiffActLoss (conv_fc -> fc -> interpolate -> refine)"

  %% Shapes legend
  subgraph Shapes[Tensor shapes]
    S1["Images: (B,T,C,H,W)"]
    S2["Image tokens: (B,T,S,token_embed_dim), S=seq_h*seq_w"]
    S3["z: (B, T*S, decoder_embed_dim)"]
    S4["History action: (B,L,act_dim) -> proj -> (B,L,embed) -> repeat-> (B, L*M,embed)"]
    S5["Text: CLIP 512 -> proj -> (B,embed) -> repeat buffer_size_text"]
  end

  Shapes --> CONCAT

  style Inputs fill:#eef,stroke:#333
  style VAE fill:#efe,stroke:#333
  style MAR_Transformer fill:#ffe,stroke:#333
  style Diff fill:#fdd,stroke:#333

  ## Key code snippets & tensor shapes

  - VAE latent extraction (shape):

    - call: `extract_latent_autoregressive(vae_model, x)`
    - file: [unified_video_action/utils/data_utils.py](unified_video_action/utils/data_utils.py#L150-L182)
    - effect: encodes `(B,T,C,H,W)` -> posterior.sample() -> `z` with shape `(B,T,Cz,Hz,Wz)` then used by MAR after `patchify` -> `(B,T,S,token_embed_dim)`

  - History action projection + repeat (关键代码片段):

    - file: [unified_video_action/model/autoregressive/mar_con_unified.py](unified_video_action/model/autoregressive/mar_con_unified.py#L420-L446)
    - snippet:

  ```py
  history_action_latents = self.history_action_proj_cond(history_nactions)
  history_action_latents_expand = history_action_latents.repeat_interleave(
      self.buffer_size_action, dim=1
  )
  ```

    - meaning: `(B,L,act_dim) -> Linear -> (B,L,embed) -> repeat_interleave(M) -> (B, L*M, embed)` to align with image tokens

  - Text (CLIP) projection + replication:

    - file: [unified_video_action/model/autoregressive/mar_con_unified.py](unified_video_action/model/autoregressive/mar_con_unified.py#L512-L526)
    - snippet:

  ```py
  text_latents = text_latents.unsqueeze(1).repeat(1, self.buffer_size_text, 1)
  text_latents = text_latents + self.text_pos_embed
  x = torch.cat([text_latents, x], dim=1)
  ```

  - Token concatenation -> proj -> Transformer (位置):

    - file: [unified_video_action/model/autoregressive/mar_con_unified.py](unified_video_action/model/autoregressive/mar_con_unified.py#L468-L506)
    - effect: concat `x, cond, history_action_latents_expand, action_latents_expand, ...` -> `self.proj_cond_x_layer(x)` -> add temporal+spatial pos emb -> encoder blocks

  - Diffusion conditioning (如何把 Transformer 输出 `z` 注入 diffusion):

    - file: [unified_video_action/model/autoregressive/diffusion_loss.py](unified_video_action/model/autoregressive/diffusion_loss.py#L80-L96)
    - snippet:

  ```py
  self.cond_embed = nn.Linear(z_channels, model_channels)
  ...
  y = t + c_emb  # t is time embedding, c_emb = cond_embed(c)
  # AdaLN modulation uses y to shift/scale residual blocks
  ```

  这些补充覆盖了 TODO 中要求的“关键代码引用与形状”。
```
