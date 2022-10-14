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


def test_make_examples():
    examples = make_examples(3, 3, 100)
    assert len(examples) == 100


def test_make_examples2():
    train_examples = make_examples(3, 3, 1)
    test_examples = make_examples(3, 3, 1, train_examples)
    assert len(train_examples) + len(test_examples) == len(
        set(train_examples) | set(test_examples)
    )
