import sys

sys.path.extend([".", "src"])
import torch
import os
import gc
from einops import rearrange
import torch.nn.functional as F
import wandb

from unified_video_action.fvd.fvd import get_fvd_logits, frechet_distance
from unified_video_action.fvd.download import load_i3d_pretrained
from unified_video_action.common.pytorch_util import dict_apply
from unified_video_action.utils.utils import AverageMeter
from unified_video_action.utils.data_utils import resize_image
from unified_video_action.utils.data_utils import (
    normalize_action,
    normalize_obs,
    unnormalize_future_action,
)
from unified_video_action.utils.data_utils import (
    process_data,
    save_image_grid,
    get_vae_latent,
    get_trajectory,
    decode_from_sample_autoregressive,
)
from unified_video_action.utils.language_model import extract_text_features




def prepare_data_predict_action(
    cfg,
    x,
    actions,
    model,
    T,
    device,
    language_goal=None,
    language_latents=None,
    eval=False,
):
    ## normalize actions and observations
    nactions = normalize_action(
        normalizer=model.normalizer,
        normalizer_type=model.normalizer_type,
        actions=actions,
    )
    x = normalize_obs(
        normalizer=model.normalizer, normalizer_type=model.normalizer_type, batch=x
    )

    ## process data
    x, proprioception_input, _ = process_data(
        x,
        task_name=cfg.task.name,
        eval=eval,
        use_proprioception=cfg.model.policy.use_proprioception,
        different_history_freq=cfg.model.policy.different_history_freq,
    )

    real, _, c, latent_size, proprioception_input = get_vae_latent(
        x, model.vae_model, eval=True, proprioception_input=proprioception_input
    )
    history_trajectory, trajectory = get_trajectory(
        nactions,
        T,
        cfg.model.policy.shift_action,
        use_history_action=cfg.model.policy.use_history_action,
    )

    text_latents = None
    if cfg.task.dataset.language_emb_model is not None:
        if "umi" in cfg.task.name:
            text_latents = language_goal
        elif language_latents is not None:
            text_latents = language_latents
        elif cfg.task.dataset.language_emb_model == "clip":
            if language_goal is None:
                raise ValueError(
                    f"{cfg.task.name} eval requires language tokens or "
                    "language_latents when language conditioning is enabled."
                )
            text_tokens = {
                "input_ids": language_goal[:, 0].long()[:, 0],
                "attention_mask": language_goal[:, 0].long()[:, 1],
            }
            text_latents = extract_text_features(
                model.text_model,
                text_tokens,
                language_emb_model=cfg.task.dataset.language_emb_model,
            )
        elif cfg.task.dataset.language_emb_model == "flant5":
            if language_goal is None:
                raise ValueError(
                    f"{cfg.task.name} eval requires language tokens or "
                    "language_latents when language conditioning is enabled."
                )
            text_tokens = language_goal[:, 0].long()
            text_latents = extract_text_features(
                model.text_model,
                text_tokens,
                language_emb_model=cfg.task.dataset.language_emb_model,
            ).float()
        else:
            raise NotImplementedError
    return (
        x,
        real,
        latent_size,
        c,
        text_latents,
        history_trajectory,
        trajectory,
        proprioception_input,
    )


def test_video_fvd(
    cfg, model, loader, it, output_dir, device, name_label="", plot_actions=False
):
    losses = dict()
    losses["fvd"] = AverageMeter()

    i3d = load_i3d_pretrained(device)
    real_embeddings = []
    pred_embeddings = []

    reals = []
    predictions = []

    n_examples = 4

    with torch.no_grad():
        for n, batch in enumerate(loader):
            if n % 10 == 0:
                print("test_video_fvd", n, len(loader))

            x = batch
            if n >= n_examples:
                break

            x = dict_apply(x, lambda x: x.to(device, non_blocking=True))
            actions = x["action"]

            if cfg.model.policy.use_history_action:
                x = dict_apply(x, lambda x: x[:, 1:])

            x = resize_image(cfg, x)

            B, T, C, H, W = x["obs"]["image"].size()
            k = min(n_examples, B)

            actions = actions[:k]
            x = dict_apply(x, lambda x: x[:k])

            language_goal = None
            language_latents = None
            if cfg.task.dataset.language_emb_model is not None:
                if "language" in x["obs"]:
                    language_goal = x["obs"]["language"]
                    del x["obs"]["language"]
                elif "language_latents" in x:
                    language_latents = x["language_latents"]
                    del x["language_latents"]
                else:
                    raise NotImplementedError

            (
                x,
                real,
                _,
                c,
                text_latents,
                history_trajectory,
                trajectory,
                proprioception_input,
            ) = prepare_data_predict_action(
                cfg,
                x,
                actions,
                model,
                T,
                device,
                language_goal=language_goal,
                language_latents=language_latents,
            )

            z, act_out = model.model.sample_tokens(
                bsz=k,
                cond=c,
                text_latents=text_latents,
                num_iter=cfg.model.policy.autoregressive_model_params.num_iter,
                cfg=cfg.model.policy.autoregressive_model_params.cfg,
                cfg_schedule=cfg.model.policy.autoregressive_model_params.cfg_schedule,
                temperature=cfg.model.policy.autoregressive_model_params.temperature,
                history_nactions=history_trajectory,
                nactions=trajectory,
                proprioception_input=proprioception_input,
                task_mode="full_dynamic_model",
            )
            pred = decode_from_sample_autoregressive(model.vae_model, z / 0.2325)
            pred = pred.clamp(-1, 1).cpu()

            pred = 1 + rearrange(pred, "(b t) c h w -> b t h w c", b=k)
            real = (1 + rearrange(real, "b c t h w -> b t h w c")).cpu()

            pred = pred * 127.5
            pred = pred.type(torch.uint8)

            real = real * 127.5
            real = real.type(torch.uint8)

            x = (1 + x) * 127.5  # b c t h w
            x = x.type(torch.uint8).cpu()

            if len(predictions) < n_examples:
                reals.append(
                    torch.cat([x[:, :, : x.size(2) // 2],rearrange(real, "b t h w c -> b c t h w"),],dim=2,))
                predictions.append(
                    torch.cat([x[:, :, : x.size(2) // 2],rearrange(pred, "b t h w c -> b c t h w"),],dim=2,))

            if real.shape[1] < 16:
                pred = pred.repeat_interleave(repeats=4, dim=1)
                real = real.repeat_interleave(repeats=4, dim=1)

            pred_embeddings.append(get_fvd_logits(pred.numpy(), i3d=i3d, device=device))
            real_embeddings.append(get_fvd_logits(real.numpy(), i3d=i3d, device=device))

    log_data = dict()
    reals = torch.cat(reals)
    predictions = torch.cat(predictions)

    real_embeddings = torch.cat(real_embeddings)
    pred_embeddings = torch.cat(pred_embeddings)
    fvd = frechet_distance(
        pred_embeddings.clone().detach(), real_embeddings.clone().detach()
    )
    fvd = fvd.item()

    os.makedirs(output_dir + "/vis", exist_ok=True)
    real_vid = save_image_grid(
        reals.cpu().numpy(),
        os.path.join(output_dir, f"vis/{name_label}real_{it}.gif"),
        drange=[0, 255],
        grid_size=(reals.size(0) // 4, 4),
    )  # [4, 3, 8, 128, 128]
    pred_vid = save_image_grid(
        predictions.cpu().numpy(),
        os.path.join(output_dir, f"vis/{name_label}predicted_{it}.gif"),
        drange=[0, 255],
        grid_size=(predictions.size(0) // 4, 4),
    )  # [4, 3, 8, 128, 128]

    real_video = wandb.Video(os.path.join(output_dir, f"vis/{name_label}real_{it}.gif"))
    pred_video = wandb.Video(os.path.join(output_dir, f"vis/{name_label}predicted_{it}.mp4"))

    log_data[f"{name_label}video_fvd"] = fvd
    log_data[f"{name_label}real_img"] = real_video
    log_data[f"{name_label}predicted_img"] = pred_video

    del i3d, real_embeddings, pred_embeddings, reals, predictions
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return log_data


def test_action_l2(
    cfg,
    model,
    loader,
    it,
    output_dir,
    device,
    text_model=None,
    name_label="",
    plot_actions=False,
):
    """
    Evaluate sampled actions in the original, unnormalized 12-D action space.

    RoboCasa action layout:
        0:3   EEF delta position
        3:6   EEF delta rotation
        6     gripper
        7     torso
        8:11  mobile base
        11    arm / base mode
    """

    device = torch.device(device)

    expected_action_dim = int(
        cfg.task.shape_meta.action.shape[0]
    )

    if expected_action_dim != 12:
        raise ValueError(
            f"Expected RoboCasa action dim 12, "
            f"but cfg contains {expected_action_dim}"
        )

    action_groups = {
        "arm_position": slice(0, 3),
        "arm_rotation": slice(3, 6),
        "gripper": slice(6, 7),
        "torso": slice(7, 8),
        "base": slice(8, 11),
        "base_mode": slice(11, 12),
    }

    group_stats = {
        name: {
            "abs_sum": 0.0,
            "sq_sum": 0.0,
            "count": 0,
        }
        for name in action_groups
    }

    overall_abs_sum = 0.0
    overall_sq_sum = 0.0
    overall_element_count = 0
    l2_sum = 0.0
    l2_count = 0

    gripper_correct = 0
    gripper_count = 0

    mode_correct = 0
    mode_count = 0

    num_batches = 0
    max_val_steps = cfg.training.get(
        "max_val_steps",
        None,
    )

    with torch.no_grad():
        for n, batch in enumerate(loader):
            if (
                max_val_steps is not None
                and n >= int(max_val_steps)
            ):
                break

            if n % 10 == 0:
                print(
                    "test_action_l2",
                    n,
                    len(loader),
                )

            x = dict_apply(
                batch,
                lambda value: value.to(
                    device,
                    non_blocking=True,
                ),
            )

            actions = x["action"]

            if cfg.model.policy.use_history_action:
                x = dict_apply(
                    x,
                    lambda value: value[:, 1:],
                )

            x = resize_image(cfg, x)

            B, T, C, H, W = x["obs"]["image"].size()

            language_goal = None
            language_latents = None

            if cfg.task.dataset.language_emb_model is not None:
                if "language" in x["obs"]:
                    language_goal = x["obs"]["language"]
                    del x["obs"]["language"]

                elif "language_latents" in x:
                    language_latents = x["language_latents"]
                    del x["language_latents"]

                else:
                    raise RuntimeError(
                        "Language-conditioned validation batch "
                        "contains neither language nor "
                        "language_latents."
                    )

            (
                x,
                real,
                _,
                c,
                text_latents,
                history_trajectory,
                trajectory,
                proprioception_input,
            ) = prepare_data_predict_action(
                cfg,
                x,
                actions,
                model,
                T,
                device,
                language_goal=language_goal,
                language_latents=language_latents,
            )

            # Diffusion sampling contains random noise.
            # Use a fixed seed for each validation batch so metrics
            # from different epochs remain directly comparable.
            rng_devices = []

            if device.type == "cuda":
                cuda_index = (
                    device.index
                    if device.index is not None
                    else torch.cuda.current_device()
                )
                rng_devices = [cuda_index]

            with torch.random.fork_rng(
                devices=rng_devices
            ):
                torch.manual_seed(12345 + n)

                _, act_out = model.model.sample_tokens(
                    bsz=B,
                    cond=c,
                    text_latents=text_latents,
                    num_iter=(
                        cfg.model.policy
                        .autoregressive_model_params
                        .num_iter
                    ),
                    cfg=(
                        cfg.model.policy
                        .autoregressive_model_params
                        .cfg
                    ),
                    cfg_schedule=(
                        cfg.model.policy
                        .autoregressive_model_params
                        .cfg_schedule
                    ),
                    temperature=(
                        cfg.model.policy
                        .autoregressive_model_params
                        .temperature
                    ),
                    history_nactions=history_trajectory,
                    nactions=trajectory,
                    proprioception_input=proprioception_input,
                    task_mode="policy_model",
                )

            if not cfg.model.policy.action_model_params.predict_action:
                continue

            pred = unnormalize_future_action(
                normalizer=model.normalizer,
                normalizer_type=model.normalizer_type,
                actions=act_out,
            )

            target = unnormalize_future_action(
                normalizer=model.normalizer,
                normalizer_type=model.normalizer_type,
                actions=trajectory,
            )

            if pred.shape[-1] != expected_action_dim:
                raise ValueError(
                    f"Predicted action dim is "
                    f"{pred.shape[-1]}, expected "
                    f"{expected_action_dim}"
                )

            if target.shape[-1] != expected_action_dim:
                raise ValueError(
                    f"Target action dim is "
                    f"{target.shape[-1]}, expected "
                    f"{expected_action_dim}"
                )

            if pred.shape[:2] != target.shape[:2]:
                raise ValueError(
                    "Predicted and target action sequence "
                    f"shapes differ: pred={pred.shape}, "
                    f"target={target.shape}"
                )

            error = pred - target

            overall_abs_sum += (
                error.abs().sum().item()
            )
            overall_sq_sum += (
                error.square().sum().item()
            )
            overall_element_count += error.numel()

            per_step_l2 = torch.linalg.vector_norm(
                error,
                ord=2,
                dim=-1,
            )

            l2_sum += per_step_l2.sum().item()
            l2_count += per_step_l2.numel()

            for group_name, group_slice in action_groups.items():
                group_error = error[..., group_slice]

                stats = group_stats[group_name]
                stats["abs_sum"] += (
                    group_error.abs().sum().item()
                )
                stats["sq_sum"] += (
                    group_error.square().sum().item()
                )
                stats["count"] += group_error.numel()

            pred_gripper_closed = pred[..., 6] < 0
            target_gripper_closed = target[..., 6] < 0

            gripper_correct += (
                pred_gripper_closed
                == target_gripper_closed
            ).sum().item()
            gripper_count += pred[..., 6].numel()

            pred_base_mode = pred[..., 11] >= 0
            target_base_mode = target[..., 11] >= 0

            mode_correct += (
                pred_base_mode
                == target_base_mode
            ).sum().item()
            mode_count += pred[..., 11].numel()

            num_batches += 1

            if cfg.training.debug:
                break

    if num_batches == 0:
        raise RuntimeError(
            "test_action_l2 processed no validation batches."
        )

    log_data = {
        f"{name_label}val_action_l2_distances": (
            l2_sum / max(l2_count, 1)
        ),
        f"{name_label}val_action_mae": (
            overall_abs_sum
            / max(overall_element_count, 1)
        ),
        f"{name_label}val_action_rmse": (
            overall_sq_sum
            / max(overall_element_count, 1)
        ) ** 0.5,
        f"{name_label}val_gripper_sign_accuracy": (
            gripper_correct
            / max(gripper_count, 1)
        ),
        f"{name_label}val_base_mode_sign_accuracy": (
            mode_correct
            / max(mode_count, 1)
        ),
    }

    for group_name, stats in group_stats.items():
        count = max(stats["count"], 1)

        log_data[
            f"{name_label}val_action_"
            f"{group_name}_mae"
        ] = stats["abs_sum"] / count

        log_data[
            f"{name_label}val_action_"
            f"{group_name}_rmse"
        ] = (
            stats["sq_sum"] / count
        ) ** 0.5

    print("\n===== validation action metrics =====")
    for key, value in log_data.items():
        print(f"{key}: {value:.6f}")

    return log_data

