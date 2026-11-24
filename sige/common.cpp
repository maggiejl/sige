#include <iostream>
#if defined(__CUDACC__) && defined(_MSC_VER)
// nvcc + MSVC cannot parse the Python/dynamo headers pulled in by torch/extension.h
// ('std': ambiguous symbol in compiled_autograd.h); the kernels only need the tensor API.
#include <torch/types.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#else
#include <torch/extension.h>
#endif

#ifdef _MSC_VER  // MSVC (also nvcc's host compiler on Windows) has no __builtin_unreachable
#define __builtin_unreachable() __assume(0)
#endif

const std::string identityName("identity"), swishName("swish");

enum OpType {
    ADD,
    MUL
};

enum ActivationType {
    IDENTITY,
    SWISH
};


inline ActivationType getActivationType(const std::string &activationName) {
    if (activationName == identityName)
        return IDENTITY;
    else if (activationName == swishName)
        return SWISH;
    else __builtin_unreachable();
}

inline bool broadcastable(const torch::Tensor &x, const torch::Tensor &y) {
    auto xSize = x.sizes();
    auto ySize = y.sizes();
    if (xSize.size() != ySize.size())
        return false;
    for (int i = 0; i < xSize.size(); ++i)
        if (xSize[i] != ySize[i] && xSize[i] != 1 && ySize[i] != 1)
            return false;
    return true;
}