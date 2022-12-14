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

from slidingpuzzle import *
from slidingpuzzle.nn import *


def test_all_examples():
    examples = all_examples(3, 3, 0, 10)
    assert len(examples) == 7, examples  # <10 b/c of duplicates
    # ensure there is at least 1 solution with size > 1
    assert any(len(y) > 1 for _, y in examples), examples


def test_random_examples():
    examples = random_examples(3, 3, 100)
    assert len(examples) == 100


def test_random_examples2():
    train_examples = tuple(ex for ex in random_examples(3, 3, 1))
    test_examples = tuple(ex for ex in random_examples(3, 3, 1, train_examples))
    assert len(train_examples) + len(test_examples) == len(
        set(train_examples) | set(test_examples)
    )
