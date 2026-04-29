1. x -> get_vae_latent() -> extract_latent_autoregressive() -> AutoencoderKL.encode() -> x,c,p
2. x,c,p -> self.model -> mar.forward() -> patchify(切块、展平) clone一份做ground_truth -> self.forward_mae_encoder() 
-> FC 根据任务灵活掩码 （inverse中cond_state被mask）-> 各组件逐通道拼接 FC 回到embed_dim-> 加上位置编码(时间+空间) 语言编码拼在前面 -> layernorm 
-> 16 Transformer Encoder Blocks -> x -> self.forward_mae_decoder() -> 加上 语言编码和位置编码(时间+空间)的拼接  -> 16 Transformer Encoder Blocks -> 去除语言编码 加上扩散位置编码(时间+空间) -> forward_loss() -> diffloss diffactloss

0. train -> workspace.run -> policy.model -> 1. and 2.-> diffloss diffactloss -> fvd actionl2(sample_token-> vae_decode) 
3. eval_sim -> envrun(policy) -> policy.predict_action -> env.step(action)  -> env.render()
4. eval_real -> policyinterference.runnode -> predict action -> mar.sampletoken

[encoder输出 tokens]
        ↓
+ text tokens（拼在前面）
        ↓
LayerNorm
        ↓
decoder（再加 position embedding）
        ↓
Transformer blocks
        ↓
去掉 text token