#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=${CHECKPOINT:-/data2/local_userdata/huxianbin/outputs/checkpoints/uva_robocasav0.2_jointmodel/checkpoints/latest.ckpt}
OUTPUT_DIR=${OUTPUT_DIR:-/data2/local_userdata/huxianbin/outputs/eval/uva_robocasav0.2_jointmodel/formal}
LOG_DIR=${LOG_DIR:-/data2/local_userdata/huxianbin/outputs/eval/uva_robocasav0.2_jointmodel/logs}
CONTROLLER_CONFIGS_PATH=${CONTROLLER_CONFIGS_PATH:-/home/huxianbin/unified_video_action/unified_video_action/config/robocasa_controller_configs.pkl}

NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-50}
SEEDS=${SEEDS:-195,196,197}
NUM_SHARDS=${NUM_SHARDS:-4}
GPU_IDS_STR=${GPU_IDS:-"4 5 6 7"}
read -r -a GPU_IDS <<< "$GPU_IDS_STR"

# The training data used object split B. Do not leave this variable undefined:
# passing an empty string overrides the checkpoint config with an invalid split.
OBJ_INSTANCE_SPLIT=${OBJ_INSTANCE_SPLIT:-B}

# Save a small diagnostic set rather than all 3600 rollouts.
SAVE_VIDEO=${SAVE_VIDEO:-1}
VIDEO_DIR=${VIDEO_DIR:-$OUTPUT_DIR/videos}
VIDEO_SEEDS=${VIDEO_SEEDS:-195}
NUM_VIDEOS_PER_JOB=${NUM_VIDEOS_PER_JOB:-1}
VIDEO_FPS=${VIDEO_FPS:-10}
VIDEO_STRIDE=${VIDEO_STRIDE:-1}

# Formal default keeps the 10-step settling period used during dataset regeneration.
# NUM_OPEN_LOOP_STEPS=1 is safer for diagnosis; set it to 8 for the original formal open-loop setting.
NUM_WAIT_STEPS=${NUM_WAIT_STEPS:-10}
NUM_OPEN_LOOP_STEPS=${NUM_OPEN_LOOP_STEPS:-1}
DEBUG_ACTION_EVERY=${DEBUG_ACTION_EVERY:-25}

if (( ${#GPU_IDS[@]} < NUM_SHARDS )); then
    echo "Need at least NUM_SHARDS GPU IDs; got ${#GPU_IDS[@]} for $NUM_SHARDS shards." >&2
    exit 1
fi

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

VIDEO_ARGS=()
if [[ "$SAVE_VIDEO" == "1" ]]; then
    mkdir -p "$VIDEO_DIR"
    VIDEO_ARGS=(
        --save_video
        --video_dir "$VIDEO_DIR"
        --video_seeds "$VIDEO_SEEDS"
        --num_videos_per_job "$NUM_VIDEOS_PER_JOB"
        --video_fps "$VIDEO_FPS"
        --video_stride "$VIDEO_STRIDE"
    )
fi

OPEN_LOOP_ARGS=()
if [[ -n "$NUM_OPEN_LOOP_STEPS" ]]; then
    OPEN_LOOP_ARGS=(--num_open_loop_steps "$NUM_OPEN_LOOP_STEPS")
fi

for ((SHARD_ID=0; SHARD_ID<NUM_SHARDS; SHARD_ID++)); do
    GPU_ID=${GPU_IDS[$SHARD_ID]}
    nohup env CUDA_VISIBLE_DEVICES="$GPU_ID" \
        MUJOCO_GL=egl \
        PYOPENGL_PLATFORM=egl \
        MUJOCO_EGL_DEVICE_ID="$GPU_ID" \
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
            --num_wait_steps "$NUM_WAIT_STEPS" \
            --debug_action_every "$DEBUG_ACTION_EVERY" \
            "${OPEN_LOOP_ARGS[@]}" \
            "${VIDEO_ARGS[@]}" \
        > "$LOG_DIR/shard$SHARD_ID.log" 2>&1 &
    echo "Launched shard $SHARD_ID on physical GPU $GPU_ID"
done

echo "Launched $NUM_SHARDS shards."
echo "Videos: $VIDEO_DIR"
echo "Stop: pkill -f eval_robocasa_uva.py"