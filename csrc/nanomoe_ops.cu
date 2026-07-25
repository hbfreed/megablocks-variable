#include "indices.h"
#include <torch/extension.h>

namespace nanomoe {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Kept for downstream consumers that build their topology with the older
  // sort-based path: nanoMOE, nanoMoEchat and variable-flex-olmo all call
  // nanomoe_ops.indices_variable. It aliases megablocks::indices, which has to
  // exist anyway for megablocks_ops.indices (live via ops/topology.py for the
  // upstream sparse dMoE), so exporting it here costs nothing.
  m.def("indices_variable", &megablocks::indices, "variable-size indices construction for sparse matrix.");
  m.def("build_topology", &megablocks::build_topology, "fused variable-size block-sparse topology construction (all six arrays in one launch).");
}

}  // namespace nanomoe
