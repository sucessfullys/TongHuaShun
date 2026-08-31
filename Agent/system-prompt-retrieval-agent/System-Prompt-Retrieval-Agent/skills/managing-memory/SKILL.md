# Skill: managing-memory

Read/write/prune the 4-layer local memory system: schedule, shared rules, long memory, per-round trees.

## Inputs
- `memory_root: Path`

## Outputs
- Atomic schedule writes, CSV/MD digests, per-pair YAMLs, long memory top-50 pruning, per-round directory tree.

## Constraints
- Atomic rename-on-write for whole-file replacements.
- Single-process writer assumption.
- Never log env-var values.

## Owner module
`system_prompt_retrieval_agent.memory.manager`
