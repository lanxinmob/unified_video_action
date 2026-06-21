  #!/bin/bash

  CHECKPOINT=/data2/local_userdata/huxianbin/outputs/checkpoints/uva_robocasa_pretrain300_joint_model/checkpoints/latest.ckpt
  OUTPUT_DIR=/data2/local_userdata/huxianbin/outputs/eval/uva_robocasa_pretrain300_joint_model/formal
  LOG_DIR=/data2/local_userdata/huxianbin/outputs/eval/uva_robocasa_pretrain300_joint_model/logs
  TASK_SETS="atomic_seen composite_seen composite_unseen"
  SPLIT=pretrain
  NUM_ROLLOUTS=50
  NUM_ENVS=5
  NUM_SHARDS=4

  mkdir -p "$LOG_DIR"

  for SHARD_ID in 0 1 2 3; do
      nohup env CUDA_VISIBLE_DEVICES=$SHARD_ID \
          MUJOCO_GL=egl \
          PYOPENGL_PLATFORM=egl \
          MUJOCO_EGL_DEVICE_ID=$SHARD_ID \
          python eval_robocasa_uva.py \
              --checkpoint "$CHECKPOINT" \
              --output_dir "$OUTPUT_DIR" \
              --task_set $TASK_SETS \
              --split "$SPLIT" \
              --device cuda:0 \
              --num_rollouts "$NUM_ROLLOUTS" \
              --num_envs "$NUM_ENVS" \
              --num_shards "$NUM_SHARDS" \
              --shard_id "$SHARD_ID" \
          > "$LOG_DIR/shard$SHARD_ID.log" 2>&1 &
  done

  echo "Launched 4 shards."
  echo "Stop: pkill -f eval_robocasa_uva.py"
