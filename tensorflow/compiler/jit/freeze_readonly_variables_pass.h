#ifndef TENSORFLOW_COMPILER_JIT_FREEZE_READONLY_VARIABLES_PASS_H_
#define TENSORFLOW_COMPILER_JIT_FREEZE_READONLY_VARIABLES_PASS_H_

#include "tensorflow/core/common_runtime/optimization_registry.h"

namespace tensorflow {

class FreezeReadonlyVariablesPass : public GraphOptimizationPass {
 public:
  absl::Status Run(const GraphOptimizationPassOptions& options) override;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_FREEZE_READONLY_VARIABLES_PASS_H_
