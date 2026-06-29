  #!/bin/bash

  CHECKPOINT=/data2/local_userdata/huxianbin/outputs/checkpoints/uva_robocasa_pretrain300_joint_model/checkpoints/latest.ckpt
  OUTPUT_DIR=/data2/local_userdata/huxianbin/outputs/eval/uva_robocasa_pretrain300_joint_model/formal
  LOG_DIR=/data2/local_userdata/huxianbin/outputs/eval/uva_robocasa_pretrain300_joint_model/logs
  CONTROLLER_CONFIGS_PATH=/home/huxianbin/unified_video_action/unified_video_action/config/robocasa_controller_configs.pkl
  NUM_TRIALS_PER_TASK=50
  SEEDS=195,196,197
  NUM_SHARDS=4
  GPU_IDS=(4 5 6 7)
  OBJ_INSTANCE_SPLIT=B

  mkdir -p "$LOG_DIR"

  for SHARD_ID in 0 1 2 3; do
      GPU_ID=${GPU_IDS[$SHARD_ID]}
      nohup env CUDA_VISIBLE_DEVICES=$GPU_ID \
          MUJOCO_GL=egl \
          PYOPENGL_PLATFORM=egl \
          MUJOCO_EGL_DEVICE_ID=$GPU_ID \
          python eval_robocasa_uva.py \
              --checkpoint "$CHECKPOINT" \
              --output_dir "$OUTPUT_DIR" \
              --device cuda:0 \
              --controller_configs_path "$CONTROLLER_CONFIGS_PATH" \
              --obj_instance_split "$OBJ_INSTANCE_SPLIT" \
              --num_trials_per_task "$NUM_TRIALS_PER_TASK" \
              --seeds "$SEEDS" \
              --num_shards "$NUM_SHARDS" \
              --shard_id "$SHARD_ID" \
          > "$LOG_DIR/shard$SHARD_ID.log" 2>&1 &
  done

  echo "Launched 4 shards."
  echo "Stop: pkill -f eval_robocasa_uva.py"
