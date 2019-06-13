// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

struct Callback {
  void(ORT_API_CALL* f)(void* param) NO_EXCEPTION;
  void* param;
};

