# Copyright 2022 Stephen Dunn

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Neural networks for guiding heuristic search.
"""

from slidingpuzzle.nn.dataset import (
    SlidingPuzzleDataset,
    load_or_build_dataset,
    load_examples,
    make_examples,
    save_examples,
)

from slidingpuzzle.nn.eval import (
    accuracy,
    evaluate,
    eval_checkpoint,
)

from slidingpuzzle.nn.heuristics import (
    get_heuristic,
    make_heuristic,
    set_heuristic,
    v1_distance,
)

from slidingpuzzle.nn.models import (
    Model_v1,
    load_model,
    save_model,
)

from slidingpuzzle.nn.paths import (
    CHECKPOINT_DIR,
    DATASET_DIR,
    TENSORBOARD_DIR,
    get_checkpoint_path,
    get_examples_path,
    get_log_dir,
)

from slidingpuzzle.nn.train import (
    launch_tensorboard,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    train,
)
