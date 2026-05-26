#include "tensorflow/core/grappler/optimizers/freeze_readonly_variables_grappler.h"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"

namespace tensorflow {
namespace grappler {
namespace {

constexpr char kGrapplerCheckpointEnvVar[] =
    "TF_XLA_FREEZE_VARIABLES_GRAPPLER_CHECKPOINT";
constexpr char kMaxTensorBytesEnvVar[] = "TF_XLA_FREEZE_VARIABLES_MAX_BYTES";
constexpr char kFreezePolicyEnvVar[] = "TF_XLA_FREEZE_VARIABLES_POLICY";
constexpr char kReadPathFreezeEnvVar[] =
    "TF_XLA_FREEZE_READONLY_VARIABLES_READ_PATH";
constexpr char kParameterMaxBytesEnvVar[] =
    "TF_XLA_FREEZE_VARIABLES_PARAMETER_MAX_BYTES";
constexpr int64_t kDefaultMaxTensorBytes = 16LL * 1024 * 1024;
constexpr int64_t kDefaultParameterMaxBytes = 4LL * 1024 * 1024;

enum class FreezePolicy {
  kValuableParameters,
  kAllSafe,
};

struct FrozenVariable {
  std::string checkpoint_key;
  DataType dtype = DT_INVALID;
  int64_t estimated_bytes = -1;
  TensorProto value;
};

struct RewriteStats {
  int candidates = 0;
  int frozen_variables = 0;
  int read_path_variables = 0;
  int rewritten_reads = 0;
  int rewritten_gathers = 0;
  int rewritten_gather_nds = 0;
  int rewritten_varhandle_direct_reads = 0;
  int rewritten_varhandle_switch_reads = 0;
  int rewritten_variablev2_reads = 0;
  int rewritten_variablev2_gathers = 0;
  int ignored_var_is_initialized = 0;
  int preserved_assigns = 0;
  int skipped_missing_value = 0;
  int skipped_too_large = 0;
  int skipped_partitioned = 0;
  int skipped_unsafe = 0;
  int skipped_policy = 0;
  int skipped_unsupported = 0;
};

std::string ToLower(absl::string_view value) {
  std::string lowered(value);
  for (char& c : lowered) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return lowered;
}

bool Contains(absl::string_view value, absl::string_view needle) {
  return value.find(needle) != absl::string_view::npos;
}

FreezePolicy FreezePolicyFromEnv() {
  const char* value = std::getenv(kFreezePolicyEnvVar);
  if (value == nullptr || value[0] == '\0') {
    return FreezePolicy::kValuableParameters;
  }

  const std::string policy = ToLower(value);
  if (policy == "all_safe" || policy == "all" || policy == "legacy") {
    return FreezePolicy::kAllSafe;
  }
  if (policy == "valuable_parameters" || policy == "valuable" ||
      policy == "parameters") {
    return FreezePolicy::kValuableParameters;
  }

  LOG(WARNING) << "Ignoring invalid " << kFreezePolicyEnvVar << "=" << value
               << "; using valuable_parameters";
  return FreezePolicy::kValuableParameters;
}

absl::string_view FreezePolicyName(FreezePolicy policy) {
  switch (policy) {
    case FreezePolicy::kAllSafe:
      return "all_safe";
    case FreezePolicy::kValuableParameters:
      return "valuable_parameters";
  }
  return "valuable_parameters";
}

bool EnvFlagEnabled(absl::string_view env_var) {
  const std::string env_var_name(env_var);
  const char* value = std::getenv(env_var_name.c_str());
  if (value == nullptr || value[0] == '\0') return false;

  const std::string normalized = ToLower(value);
  return normalized == "1" || normalized == "true" || normalized == "yes" ||
         normalized == "on";
}

int64_t MaxTensorBytes() {
  const char* value = std::getenv(kMaxTensorBytesEnvVar);
  if (value == nullptr || value[0] == '\0') return kDefaultMaxTensorBytes;

  int64_t parsed = 0;
  if (!absl::SimpleAtoi(value, &parsed) || parsed < 0) {
    LOG(WARNING) << "Ignoring invalid " << kMaxTensorBytesEnvVar << "=" << value
                 << "; using " << kDefaultMaxTensorBytes;
    return kDefaultMaxTensorBytes;
  }
  return parsed;
}

int64_t ParameterMaxBytes() {
  const char* value = std::getenv(kParameterMaxBytesEnvVar);
  if (value == nullptr || value[0] == '\0') {
    return kDefaultParameterMaxBytes;
  }

  int64_t parsed = 0;
  if (!absl::SimpleAtoi(value, &parsed) || parsed < 0) {
    LOG(WARNING) << "Ignoring invalid " << kParameterMaxBytesEnvVar << "="
                 << value << "; using " << kDefaultParameterMaxBytes;
    return kDefaultParameterMaxBytes;
  }
  return parsed;
}

bool EstimateTensorBytes(DataType dtype, const TensorShape& shape,
                         int64_t* estimated_bytes) {
  const int64_t elements = shape.num_elements();
  const int dtype_size = DataTypeSize(dtype);
  if (elements < 0 || dtype_size <= 0) return false;
  if (elements > std::numeric_limits<int64_t>::max() / dtype_size) {
    *estimated_bytes = std::numeric_limits<int64_t>::max();
    return true;
  }
  *estimated_bytes = elements * dtype_size;
  return true;
}

std::string OptionalSharedName(const NodeDef& node) {
  std::string shared_name;
  if (GetNodeAttr(node, "shared_name", &shared_name).ok() &&
      !shared_name.empty()) {
    return shared_name;
  }
  return "";
}

bool IsVariableNode(const NodeDef& node) {
  return node.op() == "VarHandleOp" || node.op() == "VariableV2";
}

bool IsMutatingVariableOp(absl::string_view op) {
  return op == "Assign" || op == "AssignAdd" || op == "AssignSub" ||
         op == "AssignVariableOp" || op == "AssignAddVariableOp" ||
         op == "AssignSubVariableOp" || op == "DestroyResourceOp" ||
         absl::StartsWith(op, "ResourceApply") ||
         absl::StartsWith(op, "ResourceScatter") ||
         absl::StartsWith(op, "Scatter");
}

std::vector<std::string> ControlInputs(const NodeDef& node) {
  std::vector<std::string> inputs;
  for (const std::string& input : node.input()) {
    if (IsControlInput(input)) inputs.push_back(input);
  }
  return inputs;
}

bool FindDataInput(const NodeDef& node, int dst_input, std::string* input) {
  int data_input = 0;
  for (const std::string& candidate : node.input()) {
    if (IsControlInput(candidate)) continue;
    if (data_input == dst_input) {
      *input = candidate;
      return true;
    }
    ++data_input;
  }
  return false;
}

void SetInputs(const std::vector<std::string>& inputs, NodeDef* node) {
  node->clear_input();
  for (const std::string& input : inputs) node->add_input(input);
}

std::string TensorName(absl::string_view node_name, int output_index) {
  if (output_index == 0) return std::string(node_name);
  return strings::StrCat(node_name, ":", output_index);
}

bool SetDataInput(NodeDef* node, int dst_input, absl::string_view new_input) {
  int data_input = 0;
  for (std::string& input : *node->mutable_input()) {
    if (IsControlInput(input)) continue;
    if (data_input == dst_input) {
      input = std::string(new_input);
      return true;
    }
    ++data_input;
  }
  return false;
}

void CopyInternalAttrs(const NodeDef& from, NodeDef* to) {
  for (const auto& attr : from.attr()) {
    if (!attr.first.empty() && attr.first[0] == '_') {
      (*to->mutable_attr())[attr.first] = attr.second;
    }
  }
}

NodeDef BaseReplacementDef(const NodeDef& old_node, absl::string_view op) {
  NodeDef node_def;
  node_def.set_name(old_node.name());
  node_def.set_op(std::string(op));
  node_def.set_device(old_node.device());
  CopyInternalAttrs(old_node, &node_def);
  return node_def;
}

void AddUniqueKey(absl::string_view key, std::vector<std::string>* keys) {
  if (key.empty()) return;
  for (const std::string& existing : *keys) {
    if (absl::string_view(existing) == key) return;
  }
  keys->push_back(std::string(key));
}

void AddCheckpointKey(absl::string_view key, std::vector<std::string>* keys) {
  AddUniqueKey(key, keys);
  AddUniqueKey(strings::StrCat(key, "/.ATTRIBUTES/VARIABLE_VALUE"), keys);
}

bool HasPartSuffix(absl::string_view value, size_t part_pos) {
  const size_t part_index_pos = part_pos + strlen("/part_");
  if (part_index_pos >= value.size()) return false;
  for (size_t i = part_index_pos; i < value.size(); ++i) {
    if (value[i] < '0' || value[i] > '9') return false;
  }
  return true;
}

void AddCandidateCheckpointKeyVariants(absl::string_view key,
                                       std::vector<std::string>* keys) {
  if (key.empty()) return;

  std::vector<std::string> bases;
  bases.push_back(std::string(key));
  if (absl::StartsWith(key, "varhandle/")) {
    bases.push_back(std::string(key.substr(strlen("varhandle/"))));
  }

  for (const std::string& base : bases) {
    AddCheckpointKey(base, keys);

    const size_t part_pos = base.rfind("/part_");
    if (part_pos != std::string::npos && HasPartSuffix(base, part_pos)) {
      AddCheckpointKey(base.substr(0, part_pos), keys);
    }
  }
}

std::vector<std::string> CandidateCheckpointKeys(const NodeDef& node) {
  std::vector<std::string> keys;
  const std::string shared_name = OptionalSharedName(node);
  if (!shared_name.empty()) {
    AddCandidateCheckpointKeyVariants(shared_name, &keys);
  }
  AddCandidateCheckpointKeyVariants(node.name(), &keys);
  return keys;
}

class CheckpointTensorReader {
 public:
  explicit CheckpointTensorReader(absl::string_view checkpoint_prefix)
      : reader_(Env::Default(), checkpoint_prefix),
        max_tensor_bytes_(MaxTensorBytes()) {}

  absl::Status status() const { return reader_.status(); }

  absl::Status Lookup(const std::vector<std::string>& keys,
                      FrozenVariable* frozen) {
    for (const std::string& key : keys) {
      if (!reader_.Contains(key)) continue;

      std::vector<TensorSlice> slices;
      absl::Status status = reader_.LookupTensorSlices(key, &slices);
      if (!status.ok()) return status;
      if (!slices.empty()) {
        return errors::FailedPrecondition(
            "checkpoint tensor is partitioned; slice-aware freeze is "
            "required: ",
            key, " slices=", slices.size());
      }

      DataType dtype = DT_INVALID;
      TensorShape shape;
      status = reader_.LookupDtypeAndShape(key, &dtype, &shape);
      if (!status.ok()) return status;

      int64_t estimated_bytes = -1;
      if (EstimateTensorBytes(dtype, shape, &estimated_bytes) &&
          estimated_bytes > max_tensor_bytes_) {
        return errors::ResourceExhausted(
            "checkpoint tensor exceeds freeze byte limit: key=", key,
            " dtype=", DataTypeString(dtype), " shape=", shape.DebugString(),
            " estimated_bytes=", estimated_bytes,
            " max_bytes=", max_tensor_bytes_);
      }

      Tensor tensor(dtype, shape);
      status = reader_.Lookup(key, &tensor);
      if (!status.ok()) return status;

      frozen->checkpoint_key = key;
      frozen->dtype = dtype;
      frozen->estimated_bytes = estimated_bytes;
      tensor.AsProtoTensorContent(&frozen->value);
      return absl::OkStatus();
    }
    return errors::NotFound("no checkpoint tensor for variable");
  }

 private:
  BundleReader reader_;
  const int64_t max_tensor_bytes_;
};

bool AttrTypeEquals(const NodeDef& node, absl::string_view attr_name,
                    DataType expected) {
  DataType actual = DT_INVALID;
  return GetNodeAttr(node, attr_name, &actual).ok() && actual == expected;
}

bool IsFrozenValueCompatibleWithNode(const NodeDef& node,
                                     const FrozenVariable& frozen) {
  DataType node_dtype = DT_INVALID;
  if (GetNodeAttr(node, "dtype", &node_dtype).ok() &&
      node_dtype != frozen.dtype) {
    return false;
  }

  PartialTensorShape node_shape;
  if (!GetNodeAttr(node, "shape", &node_shape).ok()) {
    return true;
  }

  PartialTensorShape frozen_shape;
  if (!PartialTensorShape::BuildPartialTensorShape(frozen.value.tensor_shape(),
                                                   &frozen_shape)
           .ok()) {
    return false;
  }

  return node_shape.IsCompatibleWith(frozen_shape);
}

bool IsSafeResourceConsumer(const NodeDef& consumer,
                            const FrozenVariable& frozen) {
  if (consumer.op() == "ReadVariableOp") {
    return AttrTypeEquals(consumer, "dtype", frozen.dtype);
  }
  if (consumer.op() == "ResourceGather") {
    int32_t batch_dims = 0;
    if (!GetNodeAttr(consumer, "batch_dims", &batch_dims).ok() ||
        batch_dims != 0) {
      return false;
    }
    return AttrTypeEquals(consumer, "dtype", frozen.dtype);
  }
  if (consumer.op() == "ResourceGatherNd") {
    return AttrTypeEquals(consumer, "dtype", frozen.dtype);
  }
  return false;
}

bool ResolveInputType(const NodeDef& node, int input_index,
                      DataType* input_type) {
  const OpDef* op_def = nullptr;
  return OpRegistry::Global()->LookUpOpDef(node.op(), &op_def).ok() &&
         InputTypeForNode(node, *op_def, input_index, input_type).ok();
}

using Fanouts = absl::flat_hash_map<std::string, std::vector<std::string>>;

Fanouts BuildDataFanouts(const GraphDef& graph) {
  Fanouts fanouts;
  for (const NodeDef& node : graph.node()) {
    for (const std::string& input : node.input()) {
      if (IsControlInput(input)) continue;
      fanouts[NodeName(input)].push_back(node.name());
    }
  }
  return fanouts;
}

const NodeDef* FindNode(const GraphDef& graph, absl::string_view name) {
  for (const NodeDef& node : graph.node()) {
    if (node.name() == name) return &node;
  }
  return nullptr;
}

NodeDef* FindMutableNode(GraphDef* graph, absl::string_view name) {
  for (NodeDef& node : *graph->mutable_node()) {
    if (node.name() == name) return &node;
  }
  return nullptr;
}

bool ForEachDataInputFrom(const NodeDef& consumer, absl::string_view producer,
                          const std::function<bool(int)>& visit) {
  int data_input = 0;
  for (const std::string& input : consumer.input()) {
    if (IsControlInput(input)) continue;
    if (NodeName(input) == producer && !visit(data_input)) return false;
    ++data_input;
  }
  return true;
}

bool IsSafeToFreeze(const GraphDef& graph, const Fanouts& fanouts,
                    const NodeDef& node, const FrozenVariable& frozen) {
  if (!IsFrozenValueCompatibleWithNode(node, frozen)) {
    return false;
  }

  const auto fanout_it = fanouts.find(node.name());
  if (fanout_it == fanouts.end()) return true;

  for (const std::string& consumer_name : fanout_it->second) {
    const NodeDef* consumer = FindNode(graph, consumer_name);
    if (consumer == nullptr) return false;
    if (IsMutatingVariableOp(consumer->op())) return false;

    if (node.op() == "VarHandleOp") {
      if (!IsSafeResourceConsumer(*consumer, frozen)) return false;
      bool ok = true;
      ForEachDataInputFrom(*consumer, node.name(), [&](int dst_input) {
        if (dst_input != 0) ok = false;
        return ok;
      });
      if (!ok) return false;
    } else if (node.op() == "VariableV2") {
      bool ok = true;
      ForEachDataInputFrom(*consumer, node.name(), [&](int dst_input) {
        DataType input_type = DT_INVALID;
        if (!ResolveInputType(*consumer, dst_input, &input_type) ||
            IsRefType(input_type)) {
          ok = false;
        }
        return ok;
      });
      if (!ok) return false;
    }
  }
  return true;
}

bool IsLookupLikeConsumerOp(absl::string_view op);
bool HasLookupLikeOutputConsumer(const GraphDef& graph, const Fanouts& fanouts,
                                 const NodeDef& node);

struct ReadPathAnalysis {
  int compute_read_paths = 0;
  int varhandle_direct_reads = 0;
  int varhandle_switch_reads = 0;
  int variablev2_reads = 0;
  int variablev2_gathers = 0;
  int ignored_var_is_initialized = 0;
  int preserved_assigns = 0;
  std::string unsafe_reason;
};

struct ReadPathRewriteCandidate {
  std::string node_name;
  FrozenVariable frozen;
  ReadPathAnalysis analysis;
};

void SetUnsafeReason(absl::string_view reason, ReadPathAnalysis* analysis) {
  if (analysis != nullptr && analysis->unsafe_reason.empty()) {
    analysis->unsafe_reason = std::string(reason);
  }
}

bool IsNodeInStack(const std::vector<std::string>& stack,
                   absl::string_view node_name) {
  for (const std::string& existing : stack) {
    if (absl::string_view(existing) == node_name) return true;
  }
  return false;
}

bool ForEachDataEdgeFrom(const NodeDef& consumer, absl::string_view producer,
                         const std::function<bool(int, int)>& visit) {
  int data_input = 0;
  for (const std::string& input : consumer.input()) {
    if (IsControlInput(input)) continue;
    const TensorId tensor_id = ParseTensorName(input);
    if (tensor_id.node() == producer && !visit(data_input, tensor_id.index())) {
      return false;
    }
    ++data_input;
  }
  return true;
}

bool IsSafeResourceReadLeaf(const NodeDef& consumer,
                            const FrozenVariable& frozen) {
  return IsSafeResourceConsumer(consumer, frozen);
}

bool AnalyzeResourceSwitchChain(const GraphDef& graph, const Fanouts& fanouts,
                                const NodeDef& switch_node,
                                const FrozenVariable& frozen,
                                std::vector<std::string>* stack,
                                ReadPathAnalysis* analysis) {
  if (switch_node.op() != "Switch" ||
      !AttrTypeEquals(switch_node, "T", DT_RESOURCE)) {
    SetUnsafeReason(
        strings::StrCat("unsupported resource switch node ", switch_node.name(),
                        " op=", switch_node.op()),
        analysis);
    return false;
  }
  if (IsNodeInStack(*stack, switch_node.name())) {
    SetUnsafeReason(strings::StrCat("cycle in resource switch chain at ",
                                    switch_node.name()),
                    analysis);
    return false;
  }

  stack->push_back(switch_node.name());
  const auto fanout_it = fanouts.find(switch_node.name());
  if (fanout_it == fanouts.end()) {
    stack->pop_back();
    return true;
  }

  for (const std::string& consumer_name : fanout_it->second) {
    const NodeDef* consumer = FindNode(graph, consumer_name);
    if (consumer == nullptr) {
      SetUnsafeReason(
          strings::StrCat("missing resource switch consumer ", consumer_name),
          analysis);
      stack->pop_back();
      return false;
    }

    bool ok = true;
    ForEachDataEdgeFrom(
        *consumer, switch_node.name(), [&](int dst_input, int src_output) {
          if (src_output != 0 && src_output != 1) {
            SetUnsafeReason(
                strings::StrCat("unsupported switch output ", src_output,
                                " from ", switch_node.name()),
                analysis);
            ok = false;
            return false;
          }

          if (consumer->op() == "Switch" && dst_input == 0) {
            if (!AnalyzeResourceSwitchChain(graph, fanouts, *consumer, frozen,
                                            stack, analysis)) {
              ok = false;
              return false;
            }
            return true;
          }

          if (dst_input == 0 && IsSafeResourceReadLeaf(*consumer, frozen)) {
            ++analysis->compute_read_paths;
            ++analysis->varhandle_switch_reads;
            return true;
          }

          SetUnsafeReason(
              strings::StrCat("unsupported resource switch consumer ",
                              consumer->name(), " op=", consumer->op(),
                              " input=", dst_input),
              analysis);
          ok = false;
          return false;
        });
    if (!ok) {
      stack->pop_back();
      return false;
    }
  }
  stack->pop_back();
  return true;
}

bool AnalyzeVarHandleReadPaths(const GraphDef& graph, const Fanouts& fanouts,
                               const NodeDef& node,
                               const FrozenVariable& frozen,
                               ReadPathAnalysis* analysis) {
  const auto fanout_it = fanouts.find(node.name());
  if (fanout_it == fanouts.end()) return true;

  for (const std::string& consumer_name : fanout_it->second) {
    const NodeDef* consumer = FindNode(graph, consumer_name);
    if (consumer == nullptr) return false;

    bool ok = true;
    ForEachDataEdgeFrom(
        *consumer, node.name(), [&](int dst_input, int src_output) {
          if (src_output != 0) {
            SetUnsafeReason(
                strings::StrCat("unsupported VarHandle output ", src_output,
                                " to ", consumer->name()),
                analysis);
            ok = false;
            return false;
          }
          if (consumer->op() == "VarIsInitializedOp" && dst_input == 0) {
            ++analysis->ignored_var_is_initialized;
            return true;
          }
          if (consumer->op() == "AssignVariableOp" && dst_input == 0) {
            ++analysis->preserved_assigns;
            return true;
          }
          if (dst_input == 0 && IsSafeResourceReadLeaf(*consumer, frozen)) {
            ++analysis->compute_read_paths;
            ++analysis->varhandle_direct_reads;
            return true;
          }
          if (consumer->op() == "Switch" && dst_input == 0) {
            std::vector<std::string> stack;
            if (!AnalyzeResourceSwitchChain(graph, fanouts, *consumer, frozen,
                                            &stack, analysis)) {
              ok = false;
              return false;
            }
            return true;
          }

          SetUnsafeReason(strings::StrCat("unsupported VarHandle consumer ",
                                          consumer->name(), " op=",
                                          consumer->op(), " input=", dst_input),
                          analysis);
          ok = false;
          return false;
        });
    if (!ok) return false;
  }
  return true;
}

bool IsPreservedVariableV2Consumer(const NodeDef& consumer, int dst_input,
                                   ReadPathAnalysis* analysis) {
  if (consumer.op() == "Assign" && dst_input == 0) {
    ++analysis->preserved_assigns;
    return true;
  }
  if (consumer.op() == "Save" || consumer.op() == "SaveV2") return true;
  return false;
}

bool HasSaveOutputConsumer(const GraphDef& graph, const Fanouts& fanouts,
                           const NodeDef& node) {
  const auto fanout_it = fanouts.find(node.name());
  if (fanout_it == fanouts.end()) return false;
  for (const std::string& consumer_name : fanout_it->second) {
    const NodeDef* consumer = FindNode(graph, consumer_name);
    if (consumer != nullptr &&
        (consumer->op() == "Save" || consumer->op() == "SaveV2")) {
      return true;
    }
  }
  return false;
}

bool HasSupportedValueOutputConsumers(const GraphDef& graph,
                                      const Fanouts& fanouts,
                                      const NodeDef& node,
                                      const FrozenVariable& frozen,
                                      ReadPathAnalysis* analysis) {
  const auto fanout_it = fanouts.find(node.name());
  if (fanout_it == fanouts.end()) {
    SetUnsafeReason(
        strings::StrCat("VariableV2 read has no data outputs ", node.name()),
        analysis);
    return false;
  }

  bool has_output = false;
  for (const std::string& consumer_name : fanout_it->second) {
    const NodeDef* consumer = FindNode(graph, consumer_name);
    if (consumer == nullptr) {
      SetUnsafeReason(
          strings::StrCat("missing VariableV2 read output consumer ",
                          consumer_name),
          analysis);
      return false;
    }

    bool ok = true;
    ForEachDataEdgeFrom(
        *consumer, node.name(), [&](int dst_input, int src_output) {
          has_output = true;
          if (src_output != 0) {
            SetUnsafeReason(strings::StrCat("unsupported VariableV2 read "
                                            "output ",
                                            src_output, " from ", node.name()),
                            analysis);
            ok = false;
            return false;
          }

          if (consumer->op() == "Save" || consumer->op() == "SaveV2") {
            SetUnsafeReason(
                strings::StrCat("VariableV2 read feeds save "
                                "path ",
                                node.name(), " -> ", consumer->name()),
                analysis);
            ok = false;
            return false;
          }
          if (IsMutatingVariableOp(consumer->op())) {
            SetUnsafeReason(
                strings::StrCat("VariableV2 read feeds "
                                "mutating op ",
                                node.name(), " -> ", consumer->name(),
                                " op=", consumer->op()),
                analysis);
            ok = false;
            return false;
          }

          DataType input_type = DT_INVALID;
          if (!ResolveInputType(*consumer, dst_input, &input_type)) {
            SetUnsafeReason(
                strings::StrCat("invalid VariableV2 read "
                                "output input ",
                                consumer->name(), " op=", consumer->op(),
                                " input=", dst_input),
                analysis);
            ok = false;
            return false;
          }
          if (IsRefType(input_type) || BaseType(input_type) != frozen.dtype) {
            SetUnsafeReason(
                strings::StrCat("unsupported VariableV2 read "
                                "output type ",
                                consumer->name(), " op=", consumer->op(),
                                " input=", dst_input,
                                " dtype=", DataTypeString(input_type)),
                analysis);
            ok = false;
            return false;
          }
          return true;
        });
    if (!ok) return false;
  }

  if (!has_output) {
    SetUnsafeReason(
        strings::StrCat("VariableV2 read has no data outputs ", node.name()),
        analysis);
    return false;
  }
  return true;
}

bool IsVariableV2ReadIdentity(const NodeDef& consumer, int dst_input,
                              const FrozenVariable& frozen) {
  return consumer.op() == "Identity" && dst_input == 0 &&
         AttrTypeEquals(consumer, "T", frozen.dtype);
}

bool IsSupportedVariableV2ComputeConsumer(
    const GraphDef& graph, const Fanouts& fanouts, const NodeDef& consumer,
    int dst_input, const FrozenVariable& frozen, ReadPathAnalysis* analysis) {
  if (IsVariableV2ReadIdentity(consumer, dst_input, frozen)) {
    if (!HasSupportedValueOutputConsumers(graph, fanouts, consumer, frozen,
                                          analysis)) {
      return false;
    }
    ++analysis->compute_read_paths;
    ++analysis->variablev2_reads;
    if (HasLookupLikeOutputConsumer(graph, fanouts, consumer)) {
      ++analysis->variablev2_gathers;
    }
    return true;
  }

  if (HasSaveOutputConsumer(graph, fanouts, consumer)) {
    SetUnsafeReason(strings::StrCat("VariableV2 read feeds save path ",
                                    consumer.name(), " op=", consumer.op()),
                    analysis);
    return false;
  }

  DataType input_type = DT_INVALID;
  if (!ResolveInputType(consumer, dst_input, &input_type)) {
    SetUnsafeReason(
        strings::StrCat("invalid VariableV2 consumer input ", consumer.name(),
                        " op=", consumer.op(), " input=", dst_input),
        analysis);
    return false;
  }

  if (IsRefType(input_type) || BaseType(input_type) != frozen.dtype) {
    SetUnsafeReason(strings::StrCat("unsupported VariableV2 consumer type ",
                                    consumer.name(), " op=", consumer.op(),
                                    " input=", dst_input,
                                    " dtype=", DataTypeString(input_type)),
                    analysis);
    return false;
  }

  ++analysis->compute_read_paths;
  ++analysis->variablev2_reads;
  if (IsLookupLikeConsumerOp(consumer.op())) ++analysis->variablev2_gathers;
  return true;
}

bool AnalyzeVariableV2ReadPaths(const GraphDef& graph, const Fanouts& fanouts,
                                const NodeDef& node,
                                const FrozenVariable& frozen,
                                ReadPathAnalysis* analysis) {
  const auto fanout_it = fanouts.find(node.name());
  if (fanout_it == fanouts.end()) return true;

  for (const std::string& consumer_name : fanout_it->second) {
    const NodeDef* consumer = FindNode(graph, consumer_name);
    if (consumer == nullptr) return false;

    bool ok = true;
    ForEachDataEdgeFrom(
        *consumer, node.name(), [&](int dst_input, int src_output) {
          if (src_output != 0) {
            SetUnsafeReason(
                strings::StrCat("unsupported VariableV2 output ", src_output,
                                " to ", consumer->name()),
                analysis);
            ok = false;
            return false;
          }
          if (IsPreservedVariableV2Consumer(*consumer, dst_input, analysis)) {
            return true;
          }
          if (IsMutatingVariableOp(consumer->op())) {
            SetUnsafeReason(
                strings::StrCat("unsupported mutating VariableV2 consumer ",
                                consumer->name(), " op=", consumer->op(),
                                " input=", dst_input),
                analysis);
            ok = false;
            return false;
          }
          if (!IsSupportedVariableV2ComputeConsumer(
                  graph, fanouts, *consumer, dst_input, frozen, analysis)) {
            ok = false;
            return false;
          }
          return true;
        });
    if (!ok) return false;
  }
  return true;
}

bool IsSafeToReadPathFreeze(const GraphDef& graph, const Fanouts& fanouts,
                            const NodeDef& node, const FrozenVariable& frozen,
                            ReadPathAnalysis* analysis) {
  if (!IsFrozenValueCompatibleWithNode(node, frozen)) {
    SetUnsafeReason("checkpoint tensor is incompatible with variable node",
                    analysis);
    return false;
  }

  bool safe = false;
  if (node.op() == "VarHandleOp") {
    safe = AnalyzeVarHandleReadPaths(graph, fanouts, node, frozen, analysis);
  } else if (node.op() == "VariableV2") {
    safe = AnalyzeVariableV2ReadPaths(graph, fanouts, node, frozen, analysis);
  } else {
    SetUnsafeReason(strings::StrCat("unsupported variable op ", node.op()),
                    analysis);
    return false;
  }

  if (!safe) return false;
  if (analysis->compute_read_paths == 0) {
    SetUnsafeReason("no supported compute read path", analysis);
    return false;
  }
  return true;
}

bool IsFloatingPointDType(DataType dtype) {
  return dtype == DT_FLOAT || dtype == DT_HALF || dtype == DT_BFLOAT16 ||
         dtype == DT_DOUBLE;
}

std::string NormalizePolicyIdentifier(absl::string_view value) {
  std::string normalized = ToLower(value);
  constexpr absl::string_view kAttributeSuffix = "/.attributes/variable_value";
  if (absl::EndsWith(normalized, kAttributeSuffix)) {
    normalized.resize(normalized.size() - kAttributeSuffix.size());
  }

  const size_t part_pos = normalized.rfind("/part_");
  if (part_pos != std::string::npos && HasPartSuffix(normalized, part_pos)) {
    normalized.resize(part_pos);
  }
  return normalized;
}

bool HasValuableParameterName(absl::string_view raw_name) {
  const std::string name = NormalizePolicyIdentifier(raw_name);
  return absl::EndsWith(name, "/bias") || name == "bias" ||
         absl::EndsWith(name, "/kernel") || name == "kernel" ||
         absl::EndsWith(name, "/weight") || name == "weight" ||
         absl::EndsWith(name, "/weights") || name == "weights" ||
         absl::EndsWith(name, "/gamma") || name == "gamma" ||
         absl::EndsWith(name, "/beta") || name == "beta" ||
         absl::EndsWith(name, "/scale") || name == "scale" ||
         absl::EndsWith(name, "/offset") || name == "offset" ||
         absl::EndsWith(name, "/moving_mean") || name == "moving_mean" ||
         absl::EndsWith(name, "/moving_variance") || name == "moving_variance";
}

bool HasLookupLikeName(absl::string_view raw_name) {
  const std::string name = NormalizePolicyIdentifier(raw_name);
  return Contains(name, "embedding") || Contains(name, "emb_lookup") ||
         Contains(name, "lookup") || Contains(name, "sparse_features") ||
         Contains(name, "vocab") || Contains(name, "table") ||
         Contains(name, "hash") || Contains(name, "padding_session");
}

bool IsLookupLikeConsumerOp(absl::string_view op) {
  return op == "ResourceGather" || op == "ResourceGatherNd" || op == "Gather" ||
         op == "GatherV2" || op == "GatherNd" ||
         Contains(ToLower(op), "lookup");
}

bool IsDenseLikeConsumerOp(absl::string_view op) {
  return op == "MatMul" || op == "BatchMatMul" || op == "BatchMatMulV2" ||
         op == "Conv2D" || op == "DepthwiseConv2dNative" || op == "BiasAdd" ||
         op == "FusedBatchNorm" || op == "FusedBatchNormV3";
}

bool HasConsumerMatching(const GraphDef& graph, const Fanouts& fanouts,
                         const NodeDef& node,
                         bool (*predicate)(absl::string_view)) {
  const auto fanout_it = fanouts.find(node.name());
  if (fanout_it == fanouts.end()) return false;

  for (const std::string& consumer_name : fanout_it->second) {
    const NodeDef* consumer = FindNode(graph, consumer_name);
    if (consumer == nullptr) continue;
    if (predicate(consumer->op())) return true;

    if (consumer->op() != "ReadVariableOp") continue;
    const auto read_fanout_it = fanouts.find(consumer->name());
    if (read_fanout_it == fanouts.end()) continue;
    for (const std::string& read_consumer_name : read_fanout_it->second) {
      const NodeDef* read_consumer = FindNode(graph, read_consumer_name);
      if (read_consumer != nullptr && predicate(read_consumer->op())) {
        return true;
      }
    }
  }
  return false;
}

bool HasLookupLikeConsumer(const GraphDef& graph, const Fanouts& fanouts,
                           const NodeDef& node) {
  return HasConsumerMatching(graph, fanouts, node, IsLookupLikeConsumerOp);
}

bool HasDenseLikeConsumer(const GraphDef& graph, const Fanouts& fanouts,
                          const NodeDef& node) {
  return HasConsumerMatching(graph, fanouts, node, IsDenseLikeConsumerOp);
}

bool LooksLikeLookupTableShape(const FrozenVariable& frozen) {
  const TensorShapeProto& shape = frozen.value.tensor_shape();
  if (shape.dim_size() < 2) return false;

  const int64_t rows = shape.dim(0).size();
  const int64_t width = shape.dim(shape.dim_size() - 1).size();
  if (rows < 0 || width < 0) return false;
  return (rows >= 4096 && width <= 64) || (rows >= 10000 && width <= 256);
}

bool HasValuablePolicyName(const NodeDef& node, const FrozenVariable& frozen) {
  if (HasValuableParameterName(frozen.checkpoint_key)) return true;
  if (HasValuableParameterName(node.name())) return true;
  return HasValuableParameterName(OptionalSharedName(node));
}

bool HasLookupPolicyName(const NodeDef& node, const FrozenVariable& frozen) {
  if (HasLookupLikeName(frozen.checkpoint_key)) return true;
  if (HasLookupLikeName(node.name())) return true;
  return HasLookupLikeName(OptionalSharedName(node));
}

bool IsValuableFreezeCandidate(const GraphDef& graph, const Fanouts& fanouts,
                               const NodeDef& node,
                               const FrozenVariable& frozen,
                               int64_t parameter_max_bytes,
                               std::string* skip_reason) {
  if (!IsFloatingPointDType(frozen.dtype)) {
    *skip_reason = "non-floating dtype";
    return false;
  }
  if (frozen.estimated_bytes < 0) {
    *skip_reason = "unknown tensor size";
    return false;
  }
  if (HasLookupPolicyName(node, frozen)) {
    *skip_reason = "lookup-like name";
    return false;
  }
  if (HasLookupLikeConsumer(graph, fanouts, node)) {
    *skip_reason = "lookup-like consumer";
    return false;
  }
  if (!HasValuablePolicyName(node, frozen)) {
    *skip_reason = "not a valuable parameter name";
    return false;
  }
  if (frozen.estimated_bytes > parameter_max_bytes) {
    *skip_reason = "exceeds parameter byte limit";
    return false;
  }
  if (LooksLikeLookupTableShape(frozen) &&
      !HasDenseLikeConsumer(graph, fanouts, node)) {
    *skip_reason = "lookup-like shape";
    return false;
  }
  return true;
}

bool NodeNameExists(const GraphDef& graph, absl::string_view name);
std::string NewNodeName(const GraphDef& graph, absl::string_view base);

NodeDef ConstNodeDef(const NodeDef& node, const FrozenVariable& frozen) {
  NodeDef node_def = BaseReplacementDef(node, "Const");
  AddNodeAttr("dtype", frozen.dtype, &node_def);
  AddNodeAttr("value", frozen.value, &node_def);
  SetInputs(ControlInputs(node), &node_def);
  return node_def;
}

NodeDef FrozenConstNodeDef(const GraphDef& graph, const NodeDef& node,
                           const FrozenVariable& frozen) {
  NodeDef node_def;
  node_def.set_name(
      NewNodeName(graph, strings::StrCat(node.name(), "/frozen_const")));
  node_def.set_op("Const");
  node_def.set_device(node.device());
  CopyInternalAttrs(node, &node_def);
  AddNodeAttr("dtype", frozen.dtype, &node_def);
  AddNodeAttr("value", frozen.value, &node_def);
  SetInputs(ControlInputs(node), &node_def);
  return node_def;
}

std::string GetOrCreateFrozenConst(
    GraphDef* graph, const NodeDef& variable, const FrozenVariable& frozen,
    absl::flat_hash_map<std::string, std::string>* frozen_const_by_node_name) {
  auto it = frozen_const_by_node_name->find(variable.name());
  if (it != frozen_const_by_node_name->end()) return it->second;

  const NodeDef variable_copy = variable;
  NodeDef node_def = FrozenConstNodeDef(*graph, variable_copy, frozen);
  NodeDef* frozen_const = graph->add_node();
  *frozen_const = std::move(node_def);
  (*frozen_const_by_node_name)[variable_copy.name()] = frozen_const->name();
  return frozen_const->name();
}

absl::Status RewriteReadVariableFromSource(NodeDef* node,
                                           absl::string_view value_input,
                                           const FrozenVariable& frozen) {
  std::vector<std::string> inputs;
  inputs.push_back(std::string(value_input));
  for (const std::string& input : ControlInputs(*node)) inputs.push_back(input);

  NodeDef node_def = BaseReplacementDef(*node, "Identity");
  SetInputs(inputs, &node_def);
  AddNodeAttr("T", frozen.dtype, &node_def);
  *node = std::move(node_def);
  return absl::OkStatus();
}

absl::Status RewriteReadVariable(NodeDef* node, const FrozenVariable& frozen) {
  std::string value_input;
  if (!FindDataInput(*node, 0, &value_input)) return absl::OkStatus();
  return RewriteReadVariableFromSource(node, value_input, frozen);
}

bool NodeNameExists(const GraphDef& graph, absl::string_view name) {
  return FindNode(graph, name) != nullptr;
}

std::string NewNodeName(const GraphDef& graph, absl::string_view base) {
  std::string candidate(base);
  if (!NodeNameExists(graph, candidate)) return candidate;
  for (int i = 1;; ++i) {
    candidate = strings::StrCat(base, "_", i);
    if (!NodeNameExists(graph, candidate)) return candidate;
  }
}

absl::Status MakeAxisConst(GraphDef* graph, const NodeDef& gather_node,
                           DataType axis_dtype, std::string* axis_input) {
  Tensor axis(axis_dtype, TensorShape({}));
  if (axis_dtype == DT_INT32) {
    axis.scalar<int32>()() = 0;
  } else if (axis_dtype == DT_INT64) {
    axis.scalar<int64_t>()() = 0;
  } else {
    return errors::InvalidArgument(
        "unsupported ResourceGather Tindices dtype: ",
        DataTypeString(axis_dtype));
  }

  TensorProto axis_proto;
  axis.AsProtoTensorContent(&axis_proto);

  const std::string axis_name =
      NewNodeName(*graph, strings::StrCat(gather_node.name(), "/axis"));
  NodeDef* axis_node = graph->add_node();
  axis_node->set_name(axis_name);
  axis_node->set_op("Const");
  axis_node->set_device(gather_node.device());
  AddNodeAttr("dtype", axis_dtype, axis_node);
  AddNodeAttr("value", axis_proto, axis_node);
  *axis_input = axis_node->name();
  return absl::OkStatus();
}

absl::Status RewriteResourceGatherFromSource(GraphDef* graph,
                                             const std::string& node_name,
                                             absl::string_view params_input,
                                             const FrozenVariable& frozen) {
  NodeDef* node = FindMutableNode(graph, node_name);
  if (node == nullptr) return absl::OkStatus();

  std::string indices_input;
  if (!FindDataInput(*node, 1, &indices_input)) {
    return absl::OkStatus();
  }

  DataType indices_dtype = DT_INVALID;
  if (!GetNodeAttr(*node, "Tindices", &indices_dtype).ok()) {
    return absl::OkStatus();
  }

  const NodeDef original = *node;
  std::string axis_input;
  TF_RETURN_IF_ERROR(
      MakeAxisConst(graph, original, indices_dtype, &axis_input));

  std::vector<std::string> inputs;
  inputs.push_back(std::string(params_input));
  inputs.push_back(indices_input);
  inputs.push_back(axis_input);
  for (const std::string& input : ControlInputs(original))
    inputs.push_back(input);

  NodeDef node_def = BaseReplacementDef(original, "GatherV2");
  SetInputs(inputs, &node_def);
  AddNodeAttr("Tparams", frozen.dtype, &node_def);
  AddNodeAttr("Tindices", indices_dtype, &node_def);
  AddNodeAttr("Taxis", indices_dtype, &node_def);
  AddNodeAttr("batch_dims", 0, &node_def);

  NodeDef* node_after_axis = FindMutableNode(graph, node_name);
  if (node_after_axis == nullptr) return absl::OkStatus();
  *node_after_axis = std::move(node_def);
  return absl::OkStatus();
}

absl::Status RewriteResourceGather(GraphDef* graph,
                                   const std::string& node_name,
                                   const FrozenVariable& frozen) {
  NodeDef* node = FindMutableNode(graph, node_name);
  if (node == nullptr) return absl::OkStatus();

  std::string params_input;
  if (!FindDataInput(*node, 0, &params_input)) return absl::OkStatus();
  return RewriteResourceGatherFromSource(graph, node_name, params_input,
                                         frozen);
}

absl::Status RewriteResourceGatherNdFromSource(GraphDef* graph,
                                               const std::string& node_name,
                                               absl::string_view params_input,
                                               const FrozenVariable& frozen) {
  NodeDef* node = FindMutableNode(graph, node_name);
  if (node == nullptr) return absl::OkStatus();

  std::string indices_input;
  if (!FindDataInput(*node, 1, &indices_input)) {
    return absl::OkStatus();
  }

  DataType indices_dtype = DT_INVALID;
  if (!GetNodeAttr(*node, "Tindices", &indices_dtype).ok()) {
    return absl::OkStatus();
  }

  std::vector<std::string> inputs;
  inputs.push_back(std::string(params_input));
  inputs.push_back(indices_input);
  for (const std::string& input : ControlInputs(*node)) inputs.push_back(input);

  NodeDef node_def = BaseReplacementDef(*node, "GatherNd");
  SetInputs(inputs, &node_def);
  AddNodeAttr("Tparams", frozen.dtype, &node_def);
  AddNodeAttr("Tindices", indices_dtype, &node_def);
  *node = std::move(node_def);
  return absl::OkStatus();
}

absl::Status RewriteResourceGatherNd(GraphDef* graph,
                                     const std::string& node_name,
                                     const FrozenVariable& frozen) {
  NodeDef* node = FindMutableNode(graph, node_name);
  if (node == nullptr) return absl::OkStatus();

  std::string params_input;
  if (!FindDataInput(*node, 0, &params_input)) return absl::OkStatus();
  return RewriteResourceGatherNdFromSource(graph, node_name, params_input,
                                           frozen);
}

std::string GetOrCreateMirrorSwitch(
    GraphDef* graph, const NodeDef& resource_switch,
    absl::string_view data_input, const FrozenVariable& frozen,
    absl::flat_hash_map<std::string, std::string>* mirror_switches) {
  const NodeDef switch_copy = resource_switch;
  auto it = mirror_switches->find(switch_copy.name());
  if (it != mirror_switches->end()) return it->second;

  std::string pred_input;
  if (!FindDataInput(switch_copy, 1, &pred_input)) return "";

  NodeDef* mirror = graph->add_node();
  mirror->set_name(NewNodeName(
      *graph, strings::StrCat(switch_copy.name(), "/frozen_switch")));
  mirror->set_op("Switch");
  mirror->set_device(switch_copy.device());
  CopyInternalAttrs(switch_copy, mirror);
  std::vector<std::string> inputs;
  inputs.push_back(std::string(data_input));
  inputs.push_back(pred_input);
  for (const std::string& input : ControlInputs(switch_copy)) {
    inputs.push_back(input);
  }
  SetInputs(inputs, mirror);
  AddNodeAttr("T", frozen.dtype, mirror);

  (*mirror_switches)[switch_copy.name()] = mirror->name();
  return mirror->name();
}

absl::Status RewriteResourceSwitchChain(
    GraphDef* graph, const Fanouts& fanouts, const NodeDef& resource_switch,
    absl::string_view data_input, const FrozenVariable& frozen,
    absl::flat_hash_map<std::string, std::string>* mirror_switches,
    RewriteStats* stats, int* rewritten_paths) {
  const NodeDef switch_copy = resource_switch;
  const std::string mirror_switch = GetOrCreateMirrorSwitch(
      graph, switch_copy, data_input, frozen, mirror_switches);
  if (mirror_switch.empty()) {
    return errors::InvalidArgument("Switch missing pred input: ",
                                   switch_copy.name());
  }

  const auto fanout_it = fanouts.find(switch_copy.name());
  if (fanout_it == fanouts.end()) return absl::OkStatus();

  for (const std::string& consumer_name : fanout_it->second) {
    NodeDef* consumer = FindMutableNode(graph, consumer_name);
    if (consumer == nullptr) continue;
    const NodeDef consumer_before = *consumer;

    bool ok = true;
    ForEachDataEdgeFrom(
        consumer_before, switch_copy.name(),
        [&](int dst_input, int src_output) {
          const std::string mirror_output =
              TensorName(mirror_switch, src_output);
          if (consumer_before.op() == "Switch" && dst_input == 0) {
            if (!RewriteResourceSwitchChain(
                     graph, fanouts, consumer_before, mirror_output, frozen,
                     mirror_switches, stats, rewritten_paths)
                     .ok()) {
              ok = false;
              return false;
            }
            return true;
          }

          if (consumer_before.op() == "ReadVariableOp" && dst_input == 0) {
            NodeDef* mutable_consumer =
                FindMutableNode(graph, consumer_before.name());
            if (mutable_consumer == nullptr) {
              ok = false;
              return false;
            }
            if (!RewriteReadVariableFromSource(mutable_consumer, mirror_output,
                                               frozen)
                     .ok()) {
              ok = false;
              return false;
            }
            ++stats->rewritten_reads;
            ++stats->rewritten_varhandle_switch_reads;
            ++*rewritten_paths;
            return true;
          }

          if (consumer_before.op() == "ResourceGather" && dst_input == 0) {
            if (!RewriteResourceGatherFromSource(graph, consumer_before.name(),
                                                 mirror_output, frozen)
                     .ok()) {
              ok = false;
              return false;
            }
            ++stats->rewritten_gathers;
            ++stats->rewritten_varhandle_switch_reads;
            ++*rewritten_paths;
            return true;
          }

          if (consumer_before.op() == "ResourceGatherNd" && dst_input == 0) {
            if (!RewriteResourceGatherNdFromSource(
                     graph, consumer_before.name(), mirror_output, frozen)
                     .ok()) {
              ok = false;
              return false;
            }
            ++stats->rewritten_gather_nds;
            ++stats->rewritten_varhandle_switch_reads;
            ++*rewritten_paths;
            return true;
          }
          return true;
        });
    if (!ok) return errors::Internal("failed to rewrite resource switch chain");
  }
  return absl::OkStatus();
}

absl::Status RewriteVarHandleReadPaths(
    GraphDef* graph, const Fanouts& fanouts, const NodeDef& variable,
    absl::string_view frozen_const, const FrozenVariable& frozen,
    absl::flat_hash_map<std::string, std::string>* mirror_switches,
    RewriteStats* stats, int* rewritten_paths) {
  const auto fanout_it = fanouts.find(variable.name());
  if (fanout_it == fanouts.end()) return absl::OkStatus();

  for (const std::string& consumer_name : fanout_it->second) {
    NodeDef* consumer = FindMutableNode(graph, consumer_name);
    if (consumer == nullptr) continue;
    const NodeDef consumer_before = *consumer;

    bool ok = true;
    ForEachDataEdgeFrom(
        consumer_before, variable.name(), [&](int dst_input, int src_output) {
          if (src_output != 0) return true;
          if (consumer_before.op() == "ReadVariableOp" && dst_input == 0) {
            NodeDef* mutable_consumer =
                FindMutableNode(graph, consumer_before.name());
            if (mutable_consumer == nullptr) {
              ok = false;
              return false;
            }
            if (!RewriteReadVariableFromSource(mutable_consumer, frozen_const,
                                               frozen)
                     .ok()) {
              ok = false;
              return false;
            }
            ++stats->rewritten_reads;
            ++stats->rewritten_varhandle_direct_reads;
            ++*rewritten_paths;
            return true;
          }
          if (consumer_before.op() == "ResourceGather" && dst_input == 0) {
            if (!RewriteResourceGatherFromSource(graph, consumer_before.name(),
                                                 frozen_const, frozen)
                     .ok()) {
              ok = false;
              return false;
            }
            ++stats->rewritten_gathers;
            ++stats->rewritten_varhandle_direct_reads;
            ++*rewritten_paths;
            return true;
          }
          if (consumer_before.op() == "ResourceGatherNd" && dst_input == 0) {
            if (!RewriteResourceGatherNdFromSource(
                     graph, consumer_before.name(), frozen_const, frozen)
                     .ok()) {
              ok = false;
              return false;
            }
            ++stats->rewritten_gather_nds;
            ++stats->rewritten_varhandle_direct_reads;
            ++*rewritten_paths;
            return true;
          }
          if (consumer_before.op() == "Switch" && dst_input == 0) {
            if (!RewriteResourceSwitchChain(
                     graph, fanouts, consumer_before, frozen_const, frozen,
                     mirror_switches, stats, rewritten_paths)
                     .ok()) {
              ok = false;
              return false;
            }
            return true;
          }
          return true;
        });
    if (!ok) return errors::Internal("failed to rewrite VarHandle read path");
  }
  return absl::OkStatus();
}

bool HasLookupLikeOutputConsumer(const GraphDef& graph, const Fanouts& fanouts,
                                 const NodeDef& node) {
  const auto fanout_it = fanouts.find(node.name());
  if (fanout_it == fanouts.end()) return false;
  for (const std::string& consumer_name : fanout_it->second) {
    const NodeDef* consumer = FindNode(graph, consumer_name);
    if (consumer != nullptr && IsLookupLikeConsumerOp(consumer->op())) {
      return true;
    }
  }
  return false;
}

absl::Status RewriteVariableV2ReadPaths(GraphDef* graph, const Fanouts& fanouts,
                                        const NodeDef& variable,
                                        absl::string_view frozen_const,
                                        const FrozenVariable& frozen,
                                        RewriteStats* stats,
                                        int* rewritten_paths) {
  const auto fanout_it = fanouts.find(variable.name());
  if (fanout_it == fanouts.end()) return absl::OkStatus();

  for (const std::string& consumer_name : fanout_it->second) {
    NodeDef* consumer = FindMutableNode(graph, consumer_name);
    if (consumer == nullptr) continue;
    const NodeDef consumer_before = *consumer;

    bool ok = true;
    ReadPathAnalysis unused_analysis;
    ForEachDataEdgeFrom(
        consumer_before, variable.name(), [&](int dst_input, int src_output) {
          if (src_output != 0) return true;
          if (IsPreservedVariableV2Consumer(consumer_before, dst_input,
                                            &unused_analysis) ||
              IsMutatingVariableOp(consumer_before.op())) {
            return true;
          }

          if (IsVariableV2ReadIdentity(consumer_before, dst_input, frozen)) {
            const bool has_lookup_output =
                HasLookupLikeOutputConsumer(*graph, fanouts, consumer_before);
            NodeDef* mutable_consumer =
                FindMutableNode(graph, consumer_before.name());
            if (mutable_consumer == nullptr ||
                !RewriteReadVariableFromSource(mutable_consumer, frozen_const,
                                               frozen)
                     .ok()) {
              ok = false;
              return false;
            }
            ++stats->rewritten_variablev2_reads;
            if (has_lookup_output) ++stats->rewritten_variablev2_gathers;
            ++*rewritten_paths;
            return true;
          }

          DataType input_type = DT_INVALID;
          if (!ResolveInputType(consumer_before, dst_input, &input_type) ||
              IsRefType(input_type) || BaseType(input_type) != frozen.dtype) {
            return true;
          }

          NodeDef* mutable_consumer =
              FindMutableNode(graph, consumer_before.name());
          if (mutable_consumer == nullptr ||
              !SetDataInput(mutable_consumer, dst_input, frozen_const)) {
            ok = false;
            return false;
          }
          ++stats->rewritten_variablev2_reads;
          if (IsLookupLikeConsumerOp(consumer_before.op()) ||
              HasLookupLikeOutputConsumer(*graph, fanouts, consumer_before)) {
            ++stats->rewritten_variablev2_gathers;
          }
          ++*rewritten_paths;
          return true;
        });
    if (!ok) return errors::Internal("failed to rewrite VariableV2 read path");
  }
  return absl::OkStatus();
}

absl::Status RewriteReadPaths(
    GraphDef* graph, const Fanouts& fanouts, const NodeDef& variable,
    const FrozenVariable& frozen,
    absl::flat_hash_map<std::string, std::string>* frozen_const_by_node_name,
    RewriteStats* stats, int* rewritten_paths) {
  const std::string frozen_const = GetOrCreateFrozenConst(
      graph, variable, frozen, frozen_const_by_node_name);

  if (variable.op() == "VarHandleOp") {
    absl::flat_hash_map<std::string, std::string> mirror_switches;
    return RewriteVarHandleReadPaths(graph, fanouts, variable, frozen_const,
                                     frozen, &mirror_switches, stats,
                                     rewritten_paths);
  }
  if (variable.op() == "VariableV2") {
    return RewriteVariableV2ReadPaths(graph, fanouts, variable, frozen_const,
                                      frozen, stats, rewritten_paths);
  }
  return absl::OkStatus();
}

bool GrapplerCheckpointEnvEnabled() {
  const char* checkpoint_prefix = std::getenv(kGrapplerCheckpointEnvVar);
  return checkpoint_prefix != nullptr && checkpoint_prefix[0] != '\0';
}

}  // namespace

const char* FreezeReadonlyVariablesGrapplerOptimizerName() {
  return "freeze_readonly_variables_grappler";
}

bool IsFreezeReadonlyVariablesGrapplerEnabled() {
  return GrapplerCheckpointEnvEnabled();
}

std::unique_ptr<CustomGraphOptimizer>
CreateFreezeReadonlyVariablesGrapplerOptimizer() {
  return std::make_unique<FreezeReadonlyVariablesGrapplerOptimizer>();
}

string FreezeReadonlyVariablesGrapplerOptimizer::name() const {
  return FreezeReadonlyVariablesGrapplerOptimizerName();
}

absl::Status FreezeReadonlyVariablesGrapplerOptimizer::Init(
    const RewriterConfig_CustomGraphOptimizer* config) {
  const char* checkpoint_prefix = std::getenv(kGrapplerCheckpointEnvVar);
  checkpoint_prefix_ = checkpoint_prefix == nullptr ? "" : checkpoint_prefix;
  if (checkpoint_prefix_.empty()) {
    VLOG(1) << "FreezeReadonlyVariablesGrappler disabled; set "
            << kGrapplerCheckpointEnvVar << " to a checkpoint prefix";
  }
  return absl::OkStatus();
}

absl::Status FreezeReadonlyVariablesGrapplerOptimizer::Optimize(
    Cluster* cluster, const GrapplerItem& item, GraphDef* optimized_graph) {
  *optimized_graph = item.graph;
  if (checkpoint_prefix_.empty()) return absl::OkStatus();

  CheckpointTensorReader tensor_reader(checkpoint_prefix_);
  if (!tensor_reader.status().ok()) {
    LOG(WARNING)
        << "FreezeReadonlyVariablesGrappler skipped: cannot open checkpoint "
        << checkpoint_prefix_ << ": " << tensor_reader.status();
    return absl::OkStatus();
  }

  RewriteStats stats;
  const FreezePolicy freeze_policy = FreezePolicyFromEnv();
  const int64_t parameter_max_bytes = ParameterMaxBytes();
  const bool read_path_freeze = EnvFlagEnabled(kReadPathFreezeEnvVar);
  Fanouts fanouts = BuildDataFanouts(*optimized_graph);
  absl::flat_hash_map<std::string, FrozenVariable> frozen_by_node_name;
  std::vector<std::pair<std::string, FrozenVariable>> variables_to_replace;
  std::vector<ReadPathRewriteCandidate> variables_to_rewrite_read_paths;

  std::vector<std::string> node_names;
  node_names.reserve(optimized_graph->node_size());
  for (const NodeDef& node : optimized_graph->node()) {
    node_names.push_back(node.name());
  }

  for (const std::string& node_name : node_names) {
    const NodeDef* node = FindNode(*optimized_graph, node_name);
    if (node == nullptr || !IsVariableNode(*node)) continue;
    ++stats.candidates;

    FrozenVariable frozen;
    std::vector<std::string> candidate_keys = CandidateCheckpointKeys(*node);
    absl::Status lookup_status = tensor_reader.Lookup(candidate_keys, &frozen);
    if (!lookup_status.ok()) {
      if (errors::IsResourceExhausted(lookup_status)) {
        ++stats.skipped_too_large;
        VLOG(1) << "FreezeReadonlyVariablesGrappler: skip large variable "
                << node->name() << " shared_name=" << OptionalSharedName(*node)
                << " candidates=[" << absl::StrJoin(candidate_keys, ", ")
                << "]: " << lookup_status;
      } else if (errors::IsFailedPrecondition(lookup_status)) {
        ++stats.skipped_partitioned;
        VLOG(1) << "FreezeReadonlyVariablesGrappler: skip partitioned variable "
                << node->name() << " shared_name=" << OptionalSharedName(*node)
                << " candidates=[" << absl::StrJoin(candidate_keys, ", ")
                << "]: " << lookup_status;
      } else {
        ++stats.skipped_missing_value;
        VLOG(2) << "FreezeReadonlyVariablesGrappler: no checkpoint value for "
                << node->name() << " shared_name=" << OptionalSharedName(*node)
                << " candidates=[" << absl::StrJoin(candidate_keys, ", ")
                << "]: " << lookup_status;
      }
      continue;
    }

    const bool safe_to_replace =
        IsSafeToFreeze(*optimized_graph, fanouts, *node, frozen);
    ReadPathAnalysis read_path_analysis;
    const bool safe_to_rewrite_read_path =
        read_path_freeze && !safe_to_replace &&
        IsSafeToReadPathFreeze(*optimized_graph, fanouts, *node, frozen,
                               &read_path_analysis);

    if (!safe_to_replace && !safe_to_rewrite_read_path) {
      ++stats.skipped_unsafe;
      if (read_path_freeze && !read_path_analysis.unsafe_reason.empty()) {
        ++stats.skipped_unsupported;
      }
      VLOG(1) << "FreezeReadonlyVariablesGrappler: skip unsafe variable "
              << node->name() << " from checkpoint key "
              << frozen.checkpoint_key
              << (read_path_analysis.unsafe_reason.empty()
                      ? ""
                      : strings::StrCat(" reason=",
                                        read_path_analysis.unsafe_reason));
      continue;
    }

    if (freeze_policy == FreezePolicy::kValuableParameters) {
      std::string skip_reason;
      if (!IsValuableFreezeCandidate(*optimized_graph, fanouts, *node, frozen,
                                     parameter_max_bytes, &skip_reason)) {
        ++stats.skipped_policy;
        VLOG(1) << "FreezeReadonlyVariablesGrappler: skip by policy node="
                << node->name() << " shared_name=" << OptionalSharedName(*node)
                << " checkpoint_key=" << frozen.checkpoint_key
                << " reason=" << skip_reason
                << " estimated_bytes=" << frozen.estimated_bytes
                << " dtype=" << DataTypeString(frozen.dtype);
        continue;
      }
    }

    if (safe_to_replace) {
      frozen_by_node_name[node->name()] = frozen;
      variables_to_replace.push_back({node->name(), frozen});
    } else {
      stats.ignored_var_is_initialized +=
          read_path_analysis.ignored_var_is_initialized;
      stats.preserved_assigns += read_path_analysis.preserved_assigns;
      variables_to_rewrite_read_paths.push_back(
          {node->name(), frozen, read_path_analysis});
    }
  }

  for (const auto& entry : variables_to_replace) {
    NodeDef* node = FindMutableNode(optimized_graph, entry.first);
    if (node == nullptr) continue;
    const FrozenVariable& frozen = entry.second;
    LOG(INFO) << "FreezeReadonlyVariablesGrappler freezing node="
              << node->name() << " op=" << node->op()
              << " shared_name=" << OptionalSharedName(*node)
              << " checkpoint_key=" << frozen.checkpoint_key
              << " estimated_bytes=" << frozen.estimated_bytes
              << " dtype=" << DataTypeString(frozen.dtype);
    *node = ConstNodeDef(*node, frozen);
    ++stats.frozen_variables;
  }

  std::vector<std::string> consumer_node_names;
  consumer_node_names.reserve(optimized_graph->node_size());
  for (const NodeDef& node : optimized_graph->node()) {
    consumer_node_names.push_back(node.name());
  }

  for (const std::string& node_name : consumer_node_names) {
    NodeDef* node = FindMutableNode(optimized_graph, node_name);
    if (node == nullptr) continue;
    if (node->op() != "ReadVariableOp" && node->op() != "ResourceGather" &&
        node->op() != "ResourceGatherNd") {
      continue;
    }

    std::string variable_input;
    if (!FindDataInput(*node, 0, &variable_input)) continue;
    auto frozen_it = frozen_by_node_name.find(NodeName(variable_input));
    if (frozen_it == frozen_by_node_name.end()) continue;

    if (node->op() == "ReadVariableOp") {
      TF_RETURN_IF_ERROR(RewriteReadVariable(node, frozen_it->second));
      ++stats.rewritten_reads;
    } else if (node->op() == "ResourceGather") {
      TF_RETURN_IF_ERROR(
          RewriteResourceGather(optimized_graph, node_name, frozen_it->second));
      ++stats.rewritten_gathers;
    } else if (node->op() == "ResourceGatherNd") {
      TF_RETURN_IF_ERROR(RewriteResourceGatherNd(optimized_graph, node_name,
                                                 frozen_it->second));
      ++stats.rewritten_gather_nds;
    }
  }

  absl::flat_hash_map<std::string, std::string> frozen_const_by_node_name;
  for (const ReadPathRewriteCandidate& candidate :
       variables_to_rewrite_read_paths) {
    NodeDef* node = FindMutableNode(optimized_graph, candidate.node_name);
    if (node == nullptr) continue;
    const NodeDef variable = *node;
    int rewritten_paths = 0;
    TF_RETURN_IF_ERROR(
        RewriteReadPaths(optimized_graph, fanouts, variable, candidate.frozen,
                         &frozen_const_by_node_name, &stats, &rewritten_paths));
    if (rewritten_paths == 0) continue;
    ++stats.read_path_variables;
    LOG(INFO) << "FreezeReadonlyVariablesGrappler freezing read paths node="
              << variable.name() << " op=" << variable.op()
              << " shared_name=" << OptionalSharedName(variable)
              << " checkpoint_key=" << candidate.frozen.checkpoint_key
              << " estimated_bytes=" << candidate.frozen.estimated_bytes
              << " dtype=" << DataTypeString(candidate.frozen.dtype)
              << " paths=" << rewritten_paths
              << " direct_reads=" << candidate.analysis.varhandle_direct_reads
              << " switch_reads=" << candidate.analysis.varhandle_switch_reads
              << " variablev2_reads=" << candidate.analysis.variablev2_reads;
  }

  LOG(INFO)
      << "FreezeReadonlyVariablesGrappler checkpoint=" << checkpoint_prefix_
      << " policy=" << FreezePolicyName(freeze_policy)
      << " parameter_max_bytes=" << parameter_max_bytes
      << " read_path_freeze=" << read_path_freeze
      << " candidates=" << stats.candidates
      << " frozen_variables=" << stats.frozen_variables
      << " read_path_variables=" << stats.read_path_variables
      << " rewritten_reads=" << stats.rewritten_reads
      << " rewritten_gathers=" << stats.rewritten_gathers
      << " rewritten_gather_nds=" << stats.rewritten_gather_nds
      << " rewritten_varhandle_direct_reads="
      << stats.rewritten_varhandle_direct_reads
      << " rewritten_varhandle_switch_reads="
      << stats.rewritten_varhandle_switch_reads
      << " rewritten_variablev2_reads=" << stats.rewritten_variablev2_reads
      << " rewritten_variablev2_gathers=" << stats.rewritten_variablev2_gathers
      << " ignored_var_is_initialized=" << stats.ignored_var_is_initialized
      << " preserved_assigns=" << stats.preserved_assigns
      << " skipped_missing_value=" << stats.skipped_missing_value
      << " skipped_too_large=" << stats.skipped_too_large
      << " skipped_partitioned=" << stats.skipped_partitioned
      << " skipped_unsafe=" << stats.skipped_unsafe
      << " skipped_unsupported=" << stats.skipped_unsupported
      << " skipped_policy=" << stats.skipped_policy;
  return absl::OkStatus();
}

REGISTER_GRAPH_OPTIMIZER_AS(FreezeReadonlyVariablesGrapplerOptimizer,
                            "freeze_readonly_variables_grappler");

}  // namespace grappler
}  // namespace tensorflow