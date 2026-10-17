---
name: coding-standards
description: Implement or review approved repository code changes with explicit ownership and project conventions, short code comments, and design rationale recorded in OpenSpec.
---

# Coding standards

Find the task's authorized OpenSpec and read relevant repository conventions before editing. If the task has no OpenSpec/go-ahead, use [openspec](../openspec/SKILL.md); preserve authorization already given rather than restarting review.

## Structure and contracts

Keep responsibilities focused and interfaces narrow. Follow existing architecture and naming; avoid unrelated refactors. Make ownership, lifetimes, error/lifecycle behavior, and boundary validation explicit. In this project, the standalone C++ core owns communication/reduction, integrations depend on it, and platform/framework details stay behind their boundaries. Use RAII for resources and scope build requirements/warnings to targets. Apply the configured formatter.

## Comments and rationale

- Ordinary comment blocks and docstrings in new or changed code contain at most **two physical lines**. Do not split adjacent blocks or compress a paragraph into an unreadable line to evade the limit.
- Keep concise local contracts, constraints, or necessary references near code. Use clear names and structure to express ordinary behavior; omit narration that merely restates it.
- Put rationale in the task's OpenSpec `design.md`: why an algorithm, ownership model, dependency, workaround, or trade-off was chosen. A short code reference may point to that section without repeating the explanation.
- Preserve mandatory legal notices, generated content, and tool-required annotations. Preserve essential contracts by documenting them appropriately, rather than deleting information to satisfy the line count.
- Apply the rule to the change at hand. Do not rewrite unrelated existing comments or generated/vendor files as incidental cleanup.

When a decision has durable architectural or compatibility consequences, use [architecture-decision-records](../architecture-decision-records/SKILL.md) and cross-link the ADR with the OpenSpec. Avoid duplicating long rationale across documents.

## Verify the change

Use [live-verification](../live-verification/SKILL.md) for the actual changed behavior. Keep implementation, relevant checks, and evidence in the same OpenSpec task. Review the diff for scope, borrowed-storage lifetime, error paths, comment length/rationale placement, and required project formatting/checks before completion.
