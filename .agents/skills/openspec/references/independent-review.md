# Independent OpenSpec review

Use this procedure only when the user selected or explicitly requested an independent review. Check whether that request also contains a conditional implementation go-ahead; record both accurately.

Start a new reviewer with no inherited conversation. With the collaboration tool, use `spawn_agent` and `fork_turns="none"`. Otherwise use an equivalent fresh-context mechanism. If none is available, report that limitation and offer personal review; do not substitute a context-inheriting agent and call it independent.

Supply the raw task requirements, repository location, exact three artifact paths, and necessary issue/source identifiers. Do not send the author's reasoning, desired verdict, previous review conclusions, or conversation transcript. The reviewer may independently inspect repository instructions, source, roadmap, decisions/ADRs, the issue, and primary references.

Adapt this neutral prompt with concrete paths and requirements:

```text
Independently review the proposed change in this repository.
Repository: {{absolute-repository-path}}
Raw requirements: {{user-requirements-without-author-conclusions}}
Artifacts: {{proposal-path}}, {{design-path}}, {{tasks-path}}
Issue/source: {{identifier-if-applicable}}

Read the artifacts and enough project material to assess correctness, fit,
scope, missing contracts, unresolved choices, and verification. Research
relevant uncertain facts using primary sources when useful.

This review is read-only. Do not edit files, implement, install, commit,
mutate issues, or send messages to others. Return findings with severity,
file/line or section, evidence, and the change needed. Distinguish blockers
from suggestions. If there are no findings, describe coverage and limits.
Your verdict does not grant implementation approval.
```

Record the review and disposition in the task folder, adding a separate report only when useful. Revise affected artifacts, then proceed only under the user's actual go-ahead or an already satisfied explicit conditional authorization. Do not require another unrelated reviewer count or approval workflow.
