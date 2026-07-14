from typing import Optional, Dict
import os


class TopKCheckpointManager:
    def __init__(
        self,
        save_dir,
        monitor_key: str,
        mode="min",
        k=1,
        format_str="epoch={epoch:03d}-train_loss={train_loss:.3f}.ckpt",
    ):
        assert mode in ["max", "min"]
        assert k >= 0

        self.save_dir = save_dir
        self.monitor_key = monitor_key
        self.mode = mode
        self.k = k
        self.format_str = format_str
        self.path_value_map = dict()

    def get_ckpt_path(self, data: Dict[str, float]) -> Optional[str]:
        if self.k == 0:
            return None

        value = data[self.monitor_key]
        ckpt_path = os.path.join(self.save_dir, self.format_str.format(**data))

        if len(self.path_value_map) < self.k:
            # under-capacity
            self.path_value_map[ckpt_path] = value
            return ckpt_path

        # At capacity, replace the oldest checkpoint among the worst-scoring
        # entries. Equal scores are eligible for replacement so ties retain the
        # most recently evaluated checkpoints.
        if self.mode == "max":
            worst_value = min(self.path_value_map.values())
            should_replace = value >= worst_value
        else:
            worst_value = max(self.path_value_map.values())
            should_replace = value <= worst_value

        if not should_replace:
            return None

        # Dictionaries preserve insertion order, so this selects the oldest
        # entry when multiple checkpoints share the same worst score.
        delete_path = next(
            path
            for path, current_value in self.path_value_map.items()
            if current_value == worst_value
        )
        del self.path_value_map[delete_path]
        self.path_value_map[ckpt_path] = value

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        if os.path.exists(delete_path):
            os.remove(delete_path)
        return ckpt_path
