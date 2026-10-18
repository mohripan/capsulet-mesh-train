# Verification: Native ring all-reduce

Date: 2026-10-18

Status: planning evidence only. Implementation checks in [tasks.md](tasks.md) have not run and remain unchecked.

## Planning baseline and inspection

Planning began at `5c259d3af9f15fa04af6d0444492fb0722c88f0a` on `main`, after M2 closure. `git status --short` showed only unrelated untracked `.agents/skills/find-skills/`, `.claude/` and `skills-lock.json`; preserve them.

Read the repository navigation, M1 focused design, Accepted ADRs/current phase and #12–#15 requirements. Inspected native config/communicator/transport/TCP headers, `src/communicator.cpp`, `src/tcp.cpp`, Linux socket helpers, v1 codec, native conformance/worker fixtures, CMake/install consumer, Python config construction, container entry point and actual `.github/workflows/core.yml`/`m1-validation.yml` coverage.

Observed constraints: current TCP connects only root/nonroots; listeners currently use root's port; the transport's operation lock rejects simultaneous send/receive calls; protocol v1 has a 48-byte header without phase/chunk metadata; Python constructors currently choose default native configuration. The proposed neighbor setup, one duplex operation and deliberate v2 upgrade address these actual boundaries. Existing short regression lanes remain push/PR; broad M1 acceptance remains nightly/manual.

Primary references inspected for uncertain transport/algorithm facts are linked in [design.md](design.md): Linux `poll`/socket readiness, Open MPI's send/receive scheduling explanation, and the authors' ring algorithm paper. These references inform the design; they are not runtime evidence for this engine.

## Planning checks

Author self-check on the PowerShell host used Python 3 from PATH for two one-off read-only checks:

- Scanned all nine changed/new Markdown files, resolving each relative Markdown link against its containing directory: 61 links resolved. All three OpenSpec artifacts and ADR 0006 have no template fields/`TBD`/`TODO` or checked implementation tasks. ADR remains Proposed; navigation and proposal consistently say review/go-ahead pending.
- Symbolically walked the design's exact partition and two-phase formulas for `p=1..9`, `N=0..65`: 594 cases passed. Each chunk contains each rank contribution exactly once after reduction, gather gives every rank every completed chunk, offsets cover the input without gaps/padding, and aggregate payload bytes equal `2*(p-1)*4*N`. This is a mathematical planning check using rank-contribution sets, not a native implementation, numerical test or process result.

`git diff --check` passed before publication. Inspected the actual regular/nightly workflow split and compared the proposed boundary to the current source. Author review explicitly covered two-rank socket aliasing, cyclic backpressure, root/neighbor startup ordering, all-empty participation, version-prefix rejection, full logical count limits, reduction order and terminal sequence behavior. No independent review or implementation authorization has occurred for this scope.

Planning publication is a docs-only commit/push to `main` under the standing owner instruction, without co-author trailers. #13 and #12 retain unchecked implementation/completion criteria and link these draft artifacts. Review selection and implementation authorization remain the next action.

## Implementation evidence

Not run. Populate actual environment/versions, exact commands, raw native/process/container results, tested source revision, numerical thresholds/errors, deadline/resource outcomes and exact-revision CI during authorized implementation. Conformance timing is a failure-bound observation, not a benchmark. Physical multi-host, native Windows and distinct-GPU claims require their own later evidence.
