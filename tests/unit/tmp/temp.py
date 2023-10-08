# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest


def test_skip():
    print("hello, test.")
    pytest.skip("This test is intentionally skipped")
    return
