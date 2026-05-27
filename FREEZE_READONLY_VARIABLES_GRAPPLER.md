# Freeze Readonly Variables Grappler Optimizer

This document describes the current implementation of
`freeze_readonly_variables_grappler`, implemented in
`tensorflow/tensorflow/core/grappler/optimizers/freeze_readonly_variables_grappler.cc`.

The optimizer rewrites readonly TensorFlow variables into constants, using tensor
values loaded from a SavedModel checkpoint. It is designed for Serving graphs
where inference variables are no longer mutated, and where exposing weights as
normal tensor constants can improve downstream graph optimization and XLA
clustering behavior.

## Runtime Controls

The optimizer is enabled only when this checkpoint prefix environment variable is
set:

```bash
export TF_XLA_FREEZE_VARIABLES_GRAPPLER_CHECKPOINT=/path/to/model/1/variables/variables
```

The value must be a TensorFlow checkpoint prefix, not a complete filename. For a
SavedModel directory such as:

```text
model/1/
  saved_model.pb
  variables/
    variables.index
    variables.data-00000-of-00001
```

the checkpoint prefix is:

```text
model/1/variables/variables
```

The maximum checkpoint tensor size is controlled by:

```bash
export TF_XLA_FREEZE_VARIABLES_MAX_BYTES=1048576
```

If the variable is unset or invalid, the implementation currently defaults to
`1 MiB`. Tensors larger than this limit are skipped before their full tensor
content is loaded.

There is no policy selector and no read-path feature flag in the current code.
The optimizer always uses all-safe behavior: every variable that passes the
safety checks is eligible, and read-path freezing is always attempted when full
variable replacement is not safe.

## High-Level Flow

`FreezeReadonlyVariablesGrapplerOptimizer::Optimize` is the main entry point.
The flow is:

1. Copy the input `GrapplerItem` graph into `optimized_graph`.
2. Return immediately if the checkpoint prefix is empty.
3. Open the checkpoint with `CheckpointTensorReader`.
4. Build a data-fanout map for the graph.
5. Scan all `VarHandleOp` and `VariableV2` nodes.
6. For each candidate, generate checkpoint key variants and load the checkpoint
   tensor value.
7. Try full variable replacement with `IsSafeToFreeze`.
8. If full replacement is not safe, try read-path freezing with
   `IsSafeToReadPathFreeze`.
9. Apply full replacements first.
10. Rewrite direct consumers of fully frozen variables.
11. Apply read-path rewrites for variables that could not be fully replaced.
12. Log rewrite and skip statistics.

The optimizer deliberately uses a fanout snapshot from the original graph. Later
rewrite phases use this snapshot to drive deterministic rewrites even after new
constant and mirror-switch nodes are inserted.

## Candidate Variables

Only these node types are considered variable candidates:

```text
VarHandleOp
VariableV2
```

All other graph nodes are ignored by the candidate scan.

Each candidate is matched against checkpoint keys derived from:

1. The node `shared_name` attribute, if present and non-empty.
2. The node name.

For each base key, the optimizer also tries:

```text
<base>
<base>/.ATTRIBUTES/VARIABLE_VALUE
```

The key generator also handles two graph/checkpoint naming differences:

```text
varhandle/<name>  -> <name>
<name>/part_N     -> <name>
```

This lets graph-only prefixes and simple single-shard suffixes map back to the
logical checkpoint tensor name.

## Checkpoint Tensor Loading

`CheckpointTensorReader::Lookup` tries candidate checkpoint keys in order. When
it finds a key, it performs these checks:

1. Reject partitioned checkpoint tensors, because this implementation is not
   slice-aware.
2. Read dtype and shape.
3. Estimate tensor size with `EstimateTensorBytes`.
4. Skip the tensor if the estimated size exceeds `TF_XLA_FREEZE_VARIABLES_MAX_BYTES`.
5. Load the tensor content and serialize it into `FrozenVariable::value`.

The loaded value is represented by `FrozenVariable`, which stores:

- `checkpoint_key`: the matched checkpoint key.
- `dtype`: the checkpoint tensor dtype.
- `estimated_bytes`: the estimated tensor size.
- `value`: the serialized `TensorProto` used by generated `Const` nodes.

## Full Variable Freezing

Full freezing replaces the original variable node itself with a `Const` node that
keeps the same node name. This path is selected only when `IsSafeToFreeze`
returns true.

The safety requirements are:

- The checkpoint tensor dtype and shape must be compatible with the variable
  node.
- No data consumer may be a mutating variable operation.
- For `VarHandleOp`, every data consumer must be a safe resource consumer on
  input 0.
- For `VariableV2`, every data consumer input must resolve to a non-ref input
  type.

Mutating operations include direct assignment operations, resource apply
operations, resource scatter operations, and destroy-resource operations.

Safe resource consumers currently include:

```text
ReadVariableOp
ResourceGather
ResourceGatherNd
```

For `ResourceGather`, only `batch_dims == 0` is supported.

When full freezing succeeds, the variable node is replaced by `ConstNodeDef`.
After that, direct resource consumers are rewritten:

```text
ReadVariableOp   -> Identity
ResourceGather   -> GatherV2
ResourceGatherNd -> GatherNd
```

## Read-Path Freezing

Read-path freezing is used when a variable cannot be fully replaced but some
compute reads can safely consume a constant value. The variable node remains in
the graph, and the optimizer creates a side `Const` node named like:

```text
<variable>/frozen_const
```

The original variable remains available for preserved assign, save, and
initialization paths. Only validated compute read paths are rewired to the
frozen const.

Read-path freezing is always enabled in the current implementation. It is still
guarded by `IsSafeToReadPathFreeze`, so unsupported or unsafe graph patterns are
skipped instead of rewritten.

### VarHandleOp Read Paths

For `VarHandleOp`, the analysis allows:

- `VarIsInitializedOp` on input 0, preserved and counted as ignored init checks.
- `AssignVariableOp` on input 0, preserved and counted as preserved assigns.
- Direct `ReadVariableOp`, `ResourceGather`, and `ResourceGatherNd` on input 0.
- `Switch<T=DT_RESOURCE>` chains whose downstream leaves are safe resource read
  consumers.

During rewrite:

- Direct `ReadVariableOp` becomes `Identity(frozen_const)`.
- Direct `ResourceGather` becomes `GatherV2(frozen_const, indices, axis=0)`.
- Direct `ResourceGatherNd` becomes `GatherNd(frozen_const, indices)`.
- Resource `Switch` chains are mirrored with value-typed `Switch` nodes that use
  the same predicate and control inputs.

### VariableV2 Read Paths

For `VariableV2`, the analysis allows:

- `Assign` on input 0, preserved.
- `Save` and `SaveV2`, preserved.
- Compute consumers whose input type resolves to the frozen value dtype and is
  not a ref type.
- `Identity` read nodes when the identity output only feeds supported value
  consumers.

The `VariableV2 -> Identity/read` case is handled carefully. If the identity
input is ref-typed, the optimizer does not connect the const directly to that
ref input. Instead, it rewrites the identity node itself to consume the frozen
const after verifying that the identity output is only used by safe value
consumers.

## Switch Handling

Resource switch chains require special handling because a resource switch has
`T=DT_RESOURCE`, but a frozen constant has the backing tensor dtype, such as
`DT_FLOAT`.

`AnalyzeResourceSwitchChain` validates that a chain contains only resource
`Switch` nodes and supported read/gather leaves. It also detects cycles.

`GetOrCreateMirrorSwitch` creates a mirror switch for the frozen tensor value:

```text
Switch<T=DT_RESOURCE>(resource_handle, pred)
Switch<T=frozen_dtype>(frozen_const_or_value, pred)
```

`RewriteResourceSwitchChain` recursively mirrors the switch chain and rewrites
safe read/gather leaves to use the corresponding value switch output.

## Rewrite Rules

The optimizer uses these node-level rewrites:

```text
VariableV2 / VarHandleOp -> Const              (full freeze only)
ReadVariableOp           -> Identity
ResourceGather           -> GatherV2
ResourceGatherNd         -> GatherNd
Switch<T=DT_RESOURCE>    -> Switch<T=frozen_dtype> mirror nodes
```

Generated replacement nodes preserve the original node name where the original
node is replaced in place. Newly inserted const, axis, and mirror-switch nodes
use generated names that avoid collisions.

Control inputs are preserved on replacement nodes where applicable.

## Skip Conditions

A candidate can be skipped for these reasons:

- No checkpoint value matches the generated key variants.
- The checkpoint tensor is partitioned.
- The checkpoint tensor exceeds `TF_XLA_FREEZE_VARIABLES_MAX_BYTES`.
- The checkpoint tensor dtype or shape is incompatible with the variable node.
- Full freezing finds a mutating or unsupported consumer.
- Read-path freezing finds no supported compute read path.
- Read-path freezing finds an unsupported consumer, ref-typed input, dtype
  mismatch, save path, mutating path, malformed switch chain, or cycle.

The optimizer does not use variable-name heuristics such as `weight`, `bias`, or
`kernel`. Eligibility is based on safety checks and the checkpoint tensor size
limit.

## Important Helper Functions

### Environment and checkpoint helpers

- `MaxTensorBytes`: reads `TF_XLA_FREEZE_VARIABLES_MAX_BYTES` and falls back to
  the 1 MiB default.
- `CandidateCheckpointKeys`: builds the ordered list of checkpoint key variants
  for a variable node.
- `CheckpointTensorReader::Lookup`: finds, validates, size-checks, and loads a
  checkpoint tensor into `FrozenVariable`.

### Graph helpers

- `BuildDataFanouts`: builds the producer-to-consumers map used by analysis and
  rewriting.
- `FindNode` and `FindMutableNode`: look up graph nodes by name.
- `FindDataInput`, `SetDataInput`, and `ForEachDataEdgeFrom`: operate on data
  inputs while ignoring control inputs.
- `BaseReplacementDef`: creates a replacement node that keeps the old name,
  device, and internal attributes.

### Safety analysis helpers

- `IsFrozenValueCompatibleWithNode`: checks dtype and shape compatibility.
- `IsSafeToFreeze`: determines whether full node replacement is safe.
- `IsSafeToReadPathFreeze`: determines whether read-path-only freezing is safe.
- `AnalyzeVarHandleReadPaths`: validates VarHandle read paths.
- `AnalyzeVariableV2ReadPaths`: validates VariableV2 read paths.
- `AnalyzeResourceSwitchChain`: validates resource switch chains.

### Rewrite helpers

- `ConstNodeDef`: builds a same-name const replacement for full freezing.
- `GetOrCreateFrozenConst`: inserts or reuses a side const for read-path
  freezing.
- `RewriteReadVariableFromSource`: rewrites a read op into an identity of a
  frozen value.
- `RewriteResourceGatherFromSource`: rewrites resource gather to `GatherV2`.
- `RewriteResourceGatherNdFromSource`: rewrites resource gather-nd to
  `GatherNd`.
- `RewriteResourceSwitchChain`: mirrors resource switch chains as value switch
  chains.
- `RewriteReadPaths`: dispatches read-path rewrites by variable op type.

## Logged Statistics

The final log line includes:

- `candidates`: variable nodes scanned.
- `frozen_variables`: variables replaced by same-name const nodes.
- `read_path_variables`: variables whose compute read paths were rewritten.
- `rewritten_reads`: `ReadVariableOp` rewrites.
- `rewritten_gathers`: `ResourceGather` rewrites.
- `rewritten_gather_nds`: `ResourceGatherNd` rewrites.
- `rewritten_varhandle_direct_reads`: direct VarHandle read-path rewrites.
- `rewritten_varhandle_switch_reads`: VarHandle read-path rewrites through
  switch chains.
- `rewritten_variablev2_reads`: VariableV2 read-path rewrites.
- `rewritten_variablev2_gathers`: VariableV2 rewrites that feed lookup-like
  consumers.
- `ignored_var_is_initialized`: preserved VarHandle initialization checks.
- `preserved_assigns`: assign paths preserved during read-path freezing.
- `skipped_missing_value`: candidates with no matching checkpoint tensor.
- `skipped_too_large`: candidates skipped by the tensor byte limit.
- `skipped_partitioned`: candidates backed by partitioned checkpoint tensors.
- `skipped_unsafe`: candidates that failed both full freeze and read-path
  safety.
- `skipped_unsupported`: unsafe candidates with a read-path analysis reason.

The log no longer contains policy fields or a read-path enable field because
policy filtering was removed and read-path analysis is always attempted.

## Current Behavior Summary

The optimizer is conservative about graph correctness and aggressive about
candidate selection:

- It considers every `VarHandleOp` and `VariableV2` with a matching checkpoint
  tensor.
- It limits memory exposure with `TF_XLA_FREEZE_VARIABLES_MAX_BYTES`, defaulting
  to 1 MiB.
- It fully replaces variables only when all current data consumers are safe.
- It otherwise attempts read-path freezing and rewrites only validated compute
  paths.
- It preserves mutation, save, and initialization paths that are explicitly
  supported by the read-path analysis.