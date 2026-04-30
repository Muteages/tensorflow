#include "tensorflow/compiler/jit/freeze_readonly_variables_pass.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/match.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"

namespace tensorflow {
namespace {

constexpr char kCheckpointEnvVar[] = "TF_XLA_FREEZE_VARIABLES_CHECKPOINT";

struct FrozenVariable {
  std::string checkpoint_key;
  DataType dtype = DT_INVALID;
  TensorProto value;
};

struct EdgeSpec {
  Node* src = nullptr;
  int src_output = Graph::kControlSlot;
  int dst_input = Graph::kControlSlot;
};

struct RewriteStats {
  int candidates = 0;
  int frozen_variables = 0;
  int rewritten_reads = 0;
  int rewritten_gathers = 0;
  int rewritten_gather_nds = 0;
  int skipped_missing_value = 0;
  int skipped_unsafe = 0;
  int skipped_unsupported = 0;
};

std::string OptionalSharedName(const Node& node) {
  std::string shared_name;
  if (GetNodeAttr(node.def(), "shared_name", &shared_name).ok() &&
      !shared_name.empty()) {
    return shared_name;
  }
  return "";
}

bool IsVariableNode(const Node& node) {
  return node.type_string() == "VarHandleOp" ||
         node.type_string() == "VariableV2";
}

bool IsMutatingVariableOp(absl::string_view op) {
  return op == "Assign" || op == "AssignAdd" || op == "AssignSub" ||
         op == "AssignVariableOp" || op == "AssignAddVariableOp" ||
         op == "AssignSubVariableOp" || op == "DestroyResourceOp" ||
         absl::StartsWith(op, "ResourceApply") ||
         absl::StartsWith(op, "ResourceScatter") ||
         absl::StartsWith(op, "Scatter");
}

const Edge* FindDataInputEdge(const Node& node, int dst_input) {
  for (const Edge* edge : node.in_edges()) {
    if (!edge->IsControlEdge() && edge->dst_input() == dst_input) {
      return edge;
    }
  }
  return nullptr;
}

std::vector<EdgeSpec> ControlInputs(const Node& node) {
  std::vector<EdgeSpec> inputs;
  for (const Edge* edge : node.in_edges()) {
    if (edge->IsControlEdge()) {
      inputs.push_back({edge->src(), Graph::kControlSlot, Graph::kControlSlot});
    }
  }
  return inputs;
}

void AddInputToNodeDef(const EdgeSpec& edge, NodeDef* node_def) {
  Graph::AddInput(node_def, edge.src->name(), edge.src_output);
}

void CopyInternalAttrs(const NodeDef& from, NodeDef* to) {
  for (const auto& attr : from.attr()) {
    if (!attr.first.empty() && attr.first[0] == '_') {
      (*to->mutable_attr())[attr.first] = attr.second;
    }
  }
}

NodeDef BaseReplacementDef(const Node& old_node, absl::string_view op) {
  NodeDef node_def;
  node_def.set_name(old_node.name());
  node_def.set_op(std::string(op));
  node_def.set_device(old_node.requested_device());
  CopyInternalAttrs(old_node.def(), &node_def);
  return node_def;
}

absl::Status ReplaceNode(Graph* graph, Node* old_node, NodeDef new_def,
                         const std::vector<EdgeSpec>& inputs,
                         Node** new_node_out = nullptr) {
  std::vector<EdgeSpec> outputs;
  outputs.reserve(old_node->out_edges().size());
  for (const Edge* edge : old_node->out_edges()) {
    outputs.push_back({edge->dst(), edge->src_output(), edge->dst_input()});
  }

  const std::string assigned_device = old_node->assigned_device_name();
  graph->RemoveNode(old_node);

  absl::Status status;
  Node* new_node = graph->AddNode(std::move(new_def), &status);
  if (!status.ok()) return status;
  if (!assigned_device.empty()) {
    new_node->set_assigned_device_name(assigned_device);
  }

  for (const EdgeSpec& input : inputs) {
    graph->AddEdge(input.src, input.src_output, new_node, input.dst_input);
  }
  for (const EdgeSpec& output : outputs) {
    graph->AddEdge(new_node, output.src_output, output.src, output.dst_input);
  }

  if (new_node_out != nullptr) *new_node_out = new_node;
  return absl::OkStatus();
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

void AddCandidateCheckpointKeyVariants(absl::string_view key,
                                       std::vector<std::string>* keys) {
  if (key.empty()) return;

  std::vector<std::string> bases;
  bases.push_back(std::string(key));
  if (absl::StartsWith(key, "varhandle/")) {
    bases.push_back(std::string(key.substr(strlen("varhandle/"))));
  }
  if (absl::StartsWith(key, "unclustered/")) {
    bases.push_back(std::string(key.substr(strlen("unclustered/"))));
  }

  for (const std::string& base : bases) {
    AddCheckpointKey(base, keys);
    if (absl::EndsWith(base, "/part_0")) {
      AddCheckpointKey(base.substr(0, base.size() - strlen("/part_0")), keys);
    }
  }
}

std::vector<std::string> CandidateCheckpointKeys(const Node& node) {
  std::vector<std::string> keys;
  std::string shared_name;
  if (GetNodeAttr(node.def(), "shared_name", &shared_name).ok() &&
      !shared_name.empty()) {
    AddCandidateCheckpointKeyVariants(shared_name, &keys);
  }
  AddCandidateCheckpointKeyVariants(node.name(), &keys);
  return keys;
}

class CheckpointTensorReader {
 public:
  explicit CheckpointTensorReader(absl::string_view checkpoint_prefix)
      : reader_(Env::Default(), checkpoint_prefix) {}

  absl::Status status() const { return reader_.status(); }

  absl::Status Lookup(const std::vector<std::string>& keys,
                      FrozenVariable* frozen) {
    for (const std::string& key : keys) {
      if (!reader_.Contains(key)) continue;

      DataType dtype = DT_INVALID;
      TensorShape shape;
      absl::Status status = reader_.LookupDtypeAndShape(key, &dtype, &shape);
      if (!status.ok()) return status;

      Tensor tensor(dtype, shape);
      status = reader_.Lookup(key, &tensor);
      if (!status.ok()) return status;

      frozen->checkpoint_key = key;
      frozen->dtype = dtype;
      tensor.AsProtoTensorContent(&frozen->value);
      return absl::OkStatus();
    }
    return errors::NotFound("no checkpoint tensor for variable");
  }

 private:
  BundleReader reader_;
};

bool AttrTypeEquals(const Node& node, absl::string_view attr_name,
                    DataType expected) {
  DataType actual = DT_INVALID;
  return GetNodeAttr(node.def(), attr_name, &actual).ok() && actual == expected;
}

bool IsSafeResourceConsumer(const Node& consumer,
                            const FrozenVariable& frozen) {
  if (consumer.type_string() == "ReadVariableOp") {
    return AttrTypeEquals(consumer, "dtype", frozen.dtype);
  }
  if (consumer.type_string() == "ResourceGather") {
    int32_t batch_dims = 0;
    if (!GetNodeAttr(consumer.def(), "batch_dims", &batch_dims).ok() ||
        batch_dims != 0) {
      return false;
    }
    return AttrTypeEquals(consumer, "dtype", frozen.dtype);
  }
  if (consumer.type_string() == "ResourceGatherNd") {
    return AttrTypeEquals(consumer, "dtype", frozen.dtype);
  }
  return false;
}

bool IsSafeToFreeze(const Node& node, const FrozenVariable& frozen) {
  for (const Edge* edge : node.out_edges()) {
    if (edge->IsControlEdge()) continue;

    const Node& consumer = *edge->dst();
    if (IsMutatingVariableOp(consumer.type_string())) return false;

    if (node.type_string() == "VarHandleOp") {
      if (edge->dst_input() != 0 || !IsSafeResourceConsumer(consumer, frozen)) {
        return false;
      }
    } else if (node.type_string() == "VariableV2") {
      if (edge->dst_input() < 0 || edge->dst_input() >= consumer.num_inputs() ||
          IsRefType(consumer.input_type(edge->dst_input()))) {
        return false;
      }
    }
  }
  return true;
}

NodeDef ConstNodeDef(const Node& node, const FrozenVariable& frozen) {
  NodeDef node_def = BaseReplacementDef(node, "Const");
  AddNodeAttr("dtype", frozen.dtype, &node_def);
  AddNodeAttr("value", frozen.value, &node_def);
  for (const EdgeSpec& edge : ControlInputs(node)) {
    AddInputToNodeDef(edge, &node_def);
  }
  return node_def;
}

absl::Status RewriteReadVariable(Graph* graph, Node* node,
                                 const FrozenVariable& frozen) {
  const Edge* value_edge = FindDataInputEdge(*node, 0);
  if (value_edge == nullptr) return absl::OkStatus();

  std::vector<EdgeSpec> inputs;
  inputs.push_back({value_edge->src(), value_edge->src_output(), 0});
  for (const EdgeSpec& edge : ControlInputs(*node)) inputs.push_back(edge);

  NodeDef node_def = BaseReplacementDef(*node, "Identity");
  for (const EdgeSpec& edge : inputs) AddInputToNodeDef(edge, &node_def);
  AddNodeAttr("T", frozen.dtype, &node_def);
  return ReplaceNode(graph, node, std::move(node_def), inputs);
}

absl::Status MakeAxisConst(Graph* graph, const Node& gather_node,
                           DataType axis_dtype, Node** axis_node) {
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

  NodeDef axis_def;
  axis_def.set_name(
      graph->NewName(strings::StrCat(gather_node.name(), "/axis")));
  axis_def.set_op("Const");
  axis_def.set_device(gather_node.requested_device());
  AddNodeAttr("dtype", axis_dtype, &axis_def);
  AddNodeAttr("value", axis_proto, &axis_def);

  absl::Status status;
  *axis_node = graph->AddNode(std::move(axis_def), &status);
  if (!status.ok()) return status;
  if (!gather_node.assigned_device_name().empty()) {
    (*axis_node)->set_assigned_device_name(gather_node.assigned_device_name());
  }
  return absl::OkStatus();
}

absl::Status RewriteResourceGather(Graph* graph, Node* node,
                                   const FrozenVariable& frozen) {
  const Edge* params_edge = FindDataInputEdge(*node, 0);
  const Edge* indices_edge = FindDataInputEdge(*node, 1);
  if (params_edge == nullptr || indices_edge == nullptr)
    return absl::OkStatus();

  DataType indices_dtype = DT_INVALID;
  if (!GetNodeAttr(node->def(), "Tindices", &indices_dtype).ok()) {
    return absl::OkStatus();
  }

  Node* axis_node = nullptr;
  absl::Status status = MakeAxisConst(graph, *node, indices_dtype, &axis_node);
  if (!status.ok()) return status;

  std::vector<EdgeSpec> inputs;
  inputs.push_back({params_edge->src(), params_edge->src_output(), 0});
  inputs.push_back({indices_edge->src(), indices_edge->src_output(), 1});
  inputs.push_back({axis_node, 0, 2});
  for (const EdgeSpec& edge : ControlInputs(*node)) inputs.push_back(edge);

  NodeDef node_def = BaseReplacementDef(*node, "GatherV2");
  for (const EdgeSpec& edge : inputs) AddInputToNodeDef(edge, &node_def);
  AddNodeAttr("Tparams", frozen.dtype, &node_def);
  AddNodeAttr("Tindices", indices_dtype, &node_def);
  AddNodeAttr("Taxis", indices_dtype, &node_def);
  AddNodeAttr("batch_dims", 0, &node_def);
  return ReplaceNode(graph, node, std::move(node_def), inputs);
}

absl::Status RewriteResourceGatherNd(Graph* graph, Node* node,
                                     const FrozenVariable& frozen) {
  const Edge* params_edge = FindDataInputEdge(*node, 0);
  const Edge* indices_edge = FindDataInputEdge(*node, 1);
  if (params_edge == nullptr || indices_edge == nullptr)
    return absl::OkStatus();

  DataType indices_dtype = DT_INVALID;
  if (!GetNodeAttr(node->def(), "Tindices", &indices_dtype).ok()) {
    return absl::OkStatus();
  }

  std::vector<EdgeSpec> inputs;
  inputs.push_back({params_edge->src(), params_edge->src_output(), 0});
  inputs.push_back({indices_edge->src(), indices_edge->src_output(), 1});
  for (const EdgeSpec& edge : ControlInputs(*node)) inputs.push_back(edge);

  NodeDef node_def = BaseReplacementDef(*node, "GatherNd");
  for (const EdgeSpec& edge : inputs) AddInputToNodeDef(edge, &node_def);
  AddNodeAttr("Tparams", frozen.dtype, &node_def);
  AddNodeAttr("Tindices", indices_dtype, &node_def);
  return ReplaceNode(graph, node, std::move(node_def), inputs);
}

}  // namespace

absl::Status FreezeReadonlyVariablesPass::Run(
    const GraphOptimizationPassOptions& options) {
  const char* checkpoint_prefix = std::getenv(kCheckpointEnvVar);
  if (checkpoint_prefix == nullptr || checkpoint_prefix[0] == '\0') {
    VLOG(1) << "FreezeReadonlyVariablesPass disabled; set " << kCheckpointEnvVar
            << " to a checkpoint prefix to enable it";
    return absl::OkStatus();
  }

  Graph* graph = options.graph->get();
  CheckpointTensorReader tensor_reader(checkpoint_prefix);
  if (!tensor_reader.status().ok()) {
    LOG(WARNING)
        << "FreezeReadonlyVariablesPass skipped: cannot open checkpoint "
        << checkpoint_prefix << ": " << tensor_reader.status();
    return absl::OkStatus();
  }

  const std::string before_dump = DumpGraphToFile(
      "before_freeze_readonly_variables_pass", *graph, options.flib_def);
  LOG(INFO) << "FreezeReadonlyVariablesPass before dump: " << before_dump;

  RewriteStats stats;
  absl::flat_hash_map<std::string, FrozenVariable> frozen_by_node_name;
  std::vector<std::pair<Node*, FrozenVariable>> variables_to_replace;

  std::vector<Node*> nodes;
  for (Node* node : graph->op_nodes()) nodes.push_back(node);

  for (Node* node : nodes) {
    if (!IsVariableNode(*node)) continue;
    ++stats.candidates;

    FrozenVariable frozen;
    std::vector<std::string> candidate_keys = CandidateCheckpointKeys(*node);
    absl::Status lookup_status = tensor_reader.Lookup(candidate_keys, &frozen);
    if (!lookup_status.ok()) {
      ++stats.skipped_missing_value;
      VLOG(2) << "FreezeReadonlyVariablesPass: no checkpoint value for "
              << node->name() << " shared_name=" << OptionalSharedName(*node)
              << " candidates=[" << absl::StrJoin(candidate_keys, ", ")
              << "]: " << lookup_status;
      continue;
    }

    if (!IsSafeToFreeze(*node, frozen)) {
      ++stats.skipped_unsafe;
      VLOG(1) << "FreezeReadonlyVariablesPass: skip unsafe variable "
              << node->name() << " from checkpoint key "
              << frozen.checkpoint_key;
      continue;
    }

    frozen_by_node_name[node->name()] = frozen;
    variables_to_replace.push_back({node, frozen});
  }

  for (const auto& entry : variables_to_replace) {
    Node* node = entry.first;
    const FrozenVariable& frozen = entry.second;
    LOG(INFO) << "FreezeReadonlyVariablesPass freezing node=" << node->name()
              << " op=" << node->type_string()
              << " shared_name=" << OptionalSharedName(*node)
              << " checkpoint_key=" << frozen.checkpoint_key
              << " dtype=" << DataTypeString(frozen.dtype);
    std::vector<EdgeSpec> inputs = ControlInputs(*node);
    NodeDef node_def = ConstNodeDef(*node, frozen);
    absl::Status status = ReplaceNode(graph, node, std::move(node_def), inputs);
    if (!status.ok()) return status;
    ++stats.frozen_variables;
  }

  nodes.clear();
  for (Node* node : graph->op_nodes()) nodes.push_back(node);

  for (Node* node : nodes) {
    if (node->type_string() != "ReadVariableOp" &&
        node->type_string() != "ResourceGather" &&
        node->type_string() != "ResourceGatherNd") {
      continue;
    }

    const Edge* variable_edge = FindDataInputEdge(*node, 0);
    if (variable_edge == nullptr) continue;
    auto frozen_it = frozen_by_node_name.find(variable_edge->src()->name());
    if (frozen_it == frozen_by_node_name.end()) continue;

    if (node->type_string() == "ReadVariableOp") {
      absl::Status status = RewriteReadVariable(graph, node, frozen_it->second);
      if (!status.ok()) return status;
      ++stats.rewritten_reads;
    } else if (node->type_string() == "ResourceGather") {
      absl::Status status =
          RewriteResourceGather(graph, node, frozen_it->second);
      if (!status.ok()) return status;
      ++stats.rewritten_gathers;
    } else if (node->type_string() == "ResourceGatherNd") {
      absl::Status status =
          RewriteResourceGatherNd(graph, node, frozen_it->second);
      if (!status.ok()) return status;
      ++stats.rewritten_gather_nds;
    } else {
      ++stats.skipped_unsupported;
    }
  }

  LOG(INFO) << "FreezeReadonlyVariablesPass checkpoint=" << checkpoint_prefix
            << " candidates=" << stats.candidates
            << " frozen_variables=" << stats.frozen_variables
            << " rewritten_reads=" << stats.rewritten_reads
            << " rewritten_gathers=" << stats.rewritten_gathers
            << " rewritten_gather_nds=" << stats.rewritten_gather_nds
            << " skipped_missing_value=" << stats.skipped_missing_value
            << " skipped_unsafe=" << stats.skipped_unsafe;

  const std::string after_dump = DumpGraphToFile(
      "after_freeze_readonly_variables_pass", *graph, options.flib_def);
  LOG(INFO) << "FreezeReadonlyVariablesPass after dump: " << after_dump;
  return absl::OkStatus();
}

}  // namespace tensorflow
