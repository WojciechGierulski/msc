import random
import copy
import torch
import math
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from models.switch.utils import set_experts_param
from models.switch.switch_ff import SwitchFeedForward


def get_token_idx(row, col, PATCHES_IN_ROW, PATCHES_IN_COL, N_FRAMES, frame_nr, video_nr_in_batch):
    # Expected input shape [B*(W*H*T+1), D]
    seq_len = 1 + PATCHES_IN_ROW * PATCHES_IN_COL * N_FRAMES
    spatial_patch_idx = row * PATCHES_IN_ROW + col
    start_index_for_spatial_patch_block = spatial_patch_idx * N_FRAMES
    token_index = start_index_for_spatial_patch_block + frame_nr + 1  # 1 because of CLS token
    batch_offset = seq_len * video_nr_in_batch
    token_index_batch = token_index + batch_offset
    return token_index_batch


class TokensToLabels:
    def __init__(self, N_SPLITS):
        self.experts_classes = self.get_experts_classes(N_SPLITS)
        pass

    def get_experts_classes(self, N_SPLITS):
        with open("datasets/ucf101/test_train_splits/ucfTrainTestlist/testlist01.txt", "r") as file:
            lines_test = file.readlines()

        with open("datasets/ucf101/test_train_splits/ucfTrainTestlist/trainlist01.txt", "r") as file:
            lines_train = file.readlines()

        classes = list(set([int(line.strip().split(" ")[1]) for line in lines_train]))
        classes = [c - 1 for c in classes]
        n_classes = len(classes)
        splits = []

        for split_nr in range(N_SPLITS):
            if split_nr == (N_SPLITS - 1):
                split_classes = copy.deepcopy(classes)
            else:
                split_classes = random.sample(classes, k=(n_classes // N_SPLITS))
            # split_train = [line for line in lines_train if int(line.strip().split(" ")[1]) in split_classes]
            # split_test = [line for line in lines_test if int(line.strip().split(" ")[1]) in split_classes]
            # splits.append({"train": split_train, "test": split_test, "classes": split_classes})
            splits.append(split_classes)
            for c in split_classes:
                classes.remove(c)
        return splits

    def route(self, batch_size, seq_len, device, **kwargs):
        class_to_expert_idx_map = {}
        for expert_idx, classes_for_expert in enumerate(self.experts_classes):
            for cls_label in classes_for_expert:
                class_to_expert_idx_map[cls_label] = expert_idx
        routes = torch.empty(batch_size * seq_len, device=device)
        for i in range(batch_size):
            item_class = kwargs["labels"][i]
            assigned_expert_idx = class_to_expert_idx_map.get(item_class.item())
            start_token_idx = i * seq_len
            end_token_idx = (i + 1) * seq_len
            routes[start_token_idx:end_token_idx] = assigned_expert_idx
        return routes.detach().clone()


class TokensToSubDatasets:
    def __init__(self, n_experts, dataset):
        video_index = []
        video_class = []
        for idx, (_, _class) in enumerate(dataset._labeled_videos._paths_and_labels):
            video_class.append(_class - 1)
            video_index.append(idx)
        X = pd.DataFrame({"video_index": video_index})
        y = pd.DataFrame({"video_class": video_class})
        self.skf = StratifiedKFold(n_splits=n_experts)
        self.skf.get_n_splits(X, y)
        self.experts_videos = {}
        for i, (_, test_index) in enumerate(self.skf.split(X, y)):
            self.experts_videos[i] = list(X.loc[test_index, "video_index"])

    def route(self, batch_size, seq_len, device, **kwargs):
        video_idx_to_expert_idx_map = {}
        for expert_idx in self.experts_videos:
            videos_for_expert = self.experts_videos[expert_idx]
            for video_idx in videos_for_expert:
                video_idx_to_expert_idx_map[video_idx] = expert_idx
        routes = torch.empty(batch_size * seq_len, device=device)
        for i in range(batch_size):
            item_class = kwargs["video_indexes"][i]
            assigned_expert_idx = video_idx_to_expert_idx_map.get(item_class.item())
            start_token_idx = i * seq_len
            end_token_idx = (i + 1) * seq_len
            routes[start_token_idx:end_token_idx] = assigned_expert_idx
        return routes.detach().clone()


class TemporalRouter:
    def __init__(self, n_experts, PATCHES_IN_ROW, PATCHES_IN_COL, N_FRAMES):
        self.PATCHES_IN_ROW = PATCHES_IN_ROW
        self.PATCHES_IN_COL = PATCHES_IN_COL
        self.N_FRAMES = N_FRAMES
        self.n_experts = n_experts
        assert n_experts in (2, 3)
        frames_per_expert = math.ceil(N_FRAMES / n_experts)
        self.frames_boundaries = [0]
        for i in range(n_experts):
            max_bndr = min(N_FRAMES, frames_per_expert * i + frames_per_expert)
            self.frames_boundaries.append(max_bndr)
        self.saved_routes = {}  # {batch_size: routes}

    def route(self, batch_size, seq_len, device, **kwargs):
        if batch_size in self.saved_routes:
            return self.saved_routes[batch_size].detach().clone()
        routes = torch.empty(batch_size * seq_len, device=device)  # CLS tokens are left as 0 - always to first expert\
        n_matches = 0
        for b in range(batch_size):
            for frame in range(self.N_FRAMES):
                for row in range(self.PATCHES_IN_ROW):
                    for col in range(self.PATCHES_IN_COL):
                        token_idx = get_token_idx(
                            row, col, self.PATCHES_IN_ROW, self.PATCHES_IN_COL, self.N_FRAMES, frame, b
                        )
                        for i, _ in enumerate(self.frames_boundaries):
                            if frame >= self.frames_boundaries[i] and frame < self.frames_boundaries[i + 1]:
                                routes[token_idx] = i
                                n_matches += 1
                                break
        assert batch_size * seq_len - n_matches == batch_size
        self.saved_routes[batch_size] = routes
        return routes.detach().clone()


class SpatialRouterVertical:
    # Vertical Stripes so routes are: left/right
    def __init__(self, n_experts, PATCHES_IN_ROW, PATCHES_IN_COL, N_FRAMES):
        self.PATCHES_IN_ROW = PATCHES_IN_ROW
        self.PATCHES_IN_COL = PATCHES_IN_COL
        self.N_FRAMES = N_FRAMES
        self.n_experts = n_experts
        assert n_experts in (2, 3)
        cols_per_expert = math.ceil(PATCHES_IN_ROW / n_experts)
        self.cols_boundaries = [0]
        for i in range(n_experts):
            max_bndr = min(PATCHES_IN_ROW, cols_per_expert * i + cols_per_expert)
            self.cols_boundaries.append(max_bndr)
        self.saved_routes = {}  # {batch_size: routes}

    def route(self, batch_size, seq_len, device, **kwargs):
        if batch_size in self.saved_routes:
            return self.saved_routes[batch_size].detach().clone()
        routes = torch.empty(batch_size * seq_len, device=device)  # CLS tokens are left as 0 - always to first expert\
        n_matches = 0
        for b in range(batch_size):
            for frame in range(self.N_FRAMES):
                for row in range(self.PATCHES_IN_ROW):
                    for col in range(self.PATCHES_IN_COL):
                        token_idx = get_token_idx(
                            row, col, self.PATCHES_IN_ROW, self.PATCHES_IN_COL, self.N_FRAMES, frame, b
                        )
                        for i, _ in enumerate(self.cols_boundaries):
                            if col >= self.cols_boundaries[i] and col < self.cols_boundaries[i + 1]:
                                routes[token_idx] = i
                                n_matches += 1
                                break
        assert batch_size * seq_len - n_matches == batch_size
        self.saved_routes[batch_size] = routes
        return routes.detach().clone()


class SpatialRouterHorizontal:
    # Horizontal Stripes so routes are: top/down
    def __init__(self, n_experts, PATCHES_IN_ROW, PATCHES_IN_COL, N_FRAMES):
        self.PATCHES_IN_ROW = PATCHES_IN_ROW
        self.PATCHES_IN_COL = PATCHES_IN_COL
        self.N_FRAMES = N_FRAMES
        self.n_experts = n_experts
        assert n_experts in (2, 3)
        rows_per_expert = math.ceil(PATCHES_IN_COL / n_experts)
        self.rows_boundaries = [0]
        for i in range(n_experts):
            max_bndr = min(PATCHES_IN_ROW, rows_per_expert * i + rows_per_expert)
            self.rows_per_expert.append(max_bndr)
        self.saved_routes = {}  # {batch_size: routes}

    def route(self, batch_size, seq_len, device, **kwargs):
        if batch_size in self.saved_routes:
            return self.saved_routes[batch_size].detach().clone()
        routes = torch.empty(batch_size * seq_len, device=device)  # CLS tokens are left as 0 - always to first expert\
        n_matches = 0
        for b in range(batch_size):
            for frame in range(self.N_FRAMES):
                for row in range(self.PATCHES_IN_ROW):
                    for col in range(self.PATCHES_IN_COL):
                        token_idx = get_token_idx(
                            row, col, self.PATCHES_IN_ROW, self.PATCHES_IN_COL, self.N_FRAMES, frame, b
                        )
                        for i, _ in enumerate(self.rows_boundaries):
                            if row >= self.rows_boundaries[i] and row < self.rows_boundaries[i + 1]:
                                routes[token_idx] = i
                                n_matches += 1
                                break
        assert batch_size * seq_len - n_matches == batch_size
        self.saved_routes[batch_size] = routes
        return routes.detach().clone()


class SpatialRouterCentral:
    def __init__(self, n_experts, PATCHES_IN_ROW, PATCHES_IN_COL, N_FRAMES):
        self.PATCHES_IN_ROW = PATCHES_IN_ROW
        self.PATCHES_IN_COL = PATCHES_IN_COL
        self.N_FRAMES = N_FRAMES
        self.n_experts = n_experts
        assert n_experts == 2
        self.saved_routes = {}  # {batch_size: routes}

    def route(self, batch_size, seq_len, device, **kwargs):
        if batch_size in self.saved_routes[batch_size]:
            return self.saved_routes[batch_size].detach().clone()
        routes = torch.empty(batch_size * seq_len, device=device)  # CLS tokens are left as 0 - always to first expert\
        for b in range(batch_size):
            for frame in range(self.N_FRAMES):
                for row in range(self.PATCHES_IN_ROW):
                    for col in range(self.PATCHES_IN_COL):
                        token_idx = get_token_idx(
                            row, col, self.PATCHES_IN_ROW, self.PATCHES_IN_COL, self.N_FRAMES, frame, b
                        )
                        if col >= 3 and col <= 10 and row >= 3 and row <= 10:
                            routes[token_idx] = 0
                        else:
                            routes[token_idx] = 1
        self.saved_routes[batch_size] = routes
        return routes.detach().clone()


def init_model_routers(model, config, dataset):
    model = set_experts_param(model, config["switch_layers"], "route", True)
    if config["router"] == "tokens_to_labels":
        router = TokensToLabels(config["n_experts"])
    elif config["router"] == "tokens_to_subsets":
        router = TokensToSubDatasets(config["n_experts"], dataset)
    elif config["router"] == "temporal":
        router = TemporalRouter(config["n_experts"], config["row_patches"], config["col_patches"], config["frames"])
    elif config["router"] == "spatial_vertical":
        router = SpatialRouterVertical(
            config["n_experts"], config["row_patches"], config["col_patches"], config["frames"]
        )
    elif config["router"] == "spatial_horizontal":
        router = SpatialRouterHorizontal(
            config["n_experts"], config["row_patches"], config["col_patches"], config["frames"]
        )
    elif config["router"] == "spatial_center":
        router = SpatialRouterCentral(
            config["n_experts"], config["row_patches"], config["col_patches"], config["frames"]
        )
    else:
        raise (ValueError(f"Unknown router {config['router']} when pretrain experts != 0"))
    model = set_experts_param(model, config["switch_layers"], "router", router)
    return model
