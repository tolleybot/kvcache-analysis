# SGLang Radix Caching, Stage 1: Architecture Read

**Status:** complete. **Question:** per `docs/sglang-radix-research.md`, does the radix
tree's fine-grained prefix matching survive being shared across instances, or does it
degrade to block or chunk granularity at the machine edge?

**Method:** read of current SGLang source at commit
[`0d5b5ae620`](https://github.com/sgl-project/sglang/commit/0d5b5ae620) (2026-08-24),
no documentation-derived claims. File paths below are relative to the SGLang repository
root at that commit; line numbers refer to it.

**Answer in one sentence:** the radix structure never crosses the machine edge; the
shared tier is a linear chain of content-addressed pages, keyed by chained SHA-256
hashes exactly like vLLM's block scheme, and enabling it in the reference deployment
coarsens matching to 64-token pages, which is coarser than vLLM's 16-token blocks.
**Outcome (b).**

## 1. The in-instance design: one structural tree across GPU and host

RadixAttention's cache is a radix tree over token sequences
(`python/sglang/srt/mem_cache/radix_cache.py`). Matching walks the tree and is
granular to `page_size` logical units: `RadixKey.match()` documents "Result is rounded
down to `page_size`" and truncates accordingly (`radix_cache.py:181-215`), and lookups
page-align the key first (`radix_cache.py:420,447`). The server default is
`page_size=1` on CUDA (`python/sglang/srt/arg_groups/overrides.py:2540-2559`,
`_page_size_default`), so in-instance matching is token-granular by default. This is
the radix advantage in its pure form: an arbitrary-length shared prefix matches to the
exact token.

The hierarchical cache (HiCache) does not replace this structure with tiers of a
different shape. A single tree spans the GPU tier (L1) and the host-memory tier (L2):
each `TreeNode` carries both device indices (`value`) and host indices
(`host_value`), and a node whose device copy was evicted but whose host copy exists is
`evicted and backuped` (`radix_cache.py:269-274`, write path in
`hiradix_cache.py:845-868`). Eviction takes reference-count-zero leaves first
(`radix_cache.py:593`), so interior nodes, which are precisely the prefixes shared by
multiple continuations, survive longest, and under the write-through policies they are
backed to host before the device copy is dropped. Within one instance, then, the
structural design is intact through two tiers and eviction favors shared prefixes.

## 2. The boundary mechanism: structure becomes a hash chain at L3

The remote tier (L3, `--hicache-storage-backend`) is where sharing across instances
happens, and it is reached through a generic key-value interface,
`HiCacheStorage` (`python/sglang/srt/mem_cache/hicache_storage.py:150`), introduced as
the storage-layer prototype in
[PR #7704](https://github.com/sgl-project/sglang/pull/7704) (commit `9d33fcfb8e`).
Keys are not tree references. They are per-page content hashes:

- **Key derivation.** `get_hash_str(token_ids, prior_hash, page_size)`
  (`python/sglang/srt/mem_cache/utils.py:107-113`) splits a token sequence into
  `page_size`-token pages and computes, for each page, SHA-256 over the previous
  page's digest concatenated with the page's token ids
  (`python/sglang/srt/mem_cache/cpp_utils/hash_binding.cpp`, `SHA256_Update` of prior
  digest then token bytes; native pipeline from
  [PR #28287](https://github.com/sgl-project/sglang/pull/28287)). Each page's identity
  therefore encodes its entire preceding prefix. This is the same chained
  content-addressing discipline as vLLM's prefix cache, at `page_size` rather than
  16-token granularity.
- **Write path.** When a node is backed up to storage, the tree edge is flattened:
  `write_backup_storage` collects the node chain's per-page hash list
  (`compute_node_hash_values`, `utils.py:127-136`, chaining from the parent's last
  hash) and hands `(pages, hashes)` to the controller
  (`hiradix_cache.py:921-943`). What the storage tier receives is a linear sequence of
  content-addressed pages, not tree structure.
- **Read path.** A lookup against the shared tier never consults a remote tree,
  because there is none. `query_storage_hit_length` page-aligns the request tokens
  (`hiradix_cache.py:1483-1488`) and `_storage_hit_query` recomputes the chained page
  hashes from the raw tokens, then asks the backend `batch_exists` for the count of
  consecutive present pages, stopping at the first gap
  (`python/sglang/srt/managers/cache_controller.py:1103-1126`). The usable hit is
  truncated to a page multiple (`hiradix_cache.py:1515`). Cross-boundary matching is
  therefore a sequential, page-granular, content-addressed prefix probe. It is
  identical in kind to a vLLM block-hash lookup.

## 3. What the Mooncake backend adds, and requires

The Mooncake Store backend
(`python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py`) stores one
object per page, sized `ksize_per_token * page_size`
(`mooncake_store.py:693`), keyed by the page hash with an optional model-aware
namespace prefix ([PR #31920](https://github.com/sgl-project/sglang/pull/31920),
`_tag_keys`, `mooncake_store.py:715-718`). Its `batch_exists` implements the
consecutive-pages contract the controller expects (`mooncake_store.py:1257`).

Granularity in practice is set by object economics. Nothing in `server_args.py`
forces a larger page when storage is enabled, so `page_size=1` with per-token network
objects is formally expressible, but the backend's own reference deployment uses
`--page-size 64`
(`python/sglang/srt/mem_cache/storage/mooncake_store/README.md:437,452`). And because
one `page_size` governs both the tree walk and the storage keys
(`hiradix_cache.py:86`, `radix_cache.py:181-215`), choosing a page size that makes the
shared tier efficient coarsens local matching to the same 64-token unit. Enabling
distribution in the recommended configuration does not merely fail to preserve
token-level matching at the boundary; it removes it inside the instance too, landing
coarser than vLLM's 16-token blocks.

## 4. The cross-instance story upstream

There is no mechanism for sharing or synchronizing the tree itself. No broadcast,
replication, or sharding of `TreeNode` structures exists in `python/sglang/srt`
(searched at the same commit). The upstream cross-instance component is the router
(`experimental/sgl-router`), whose cache-aware policies consume KV events keyed by the
same page hashes (`experimental/sgl-router/src/policies/kv_events/hash.rs`) to route
requests toward the worker likely to hold the prefix. That is routing to structure,
not sharing of structure: each instance's tree remains private, and the only shared
artifacts are content-addressed pages in L3 and hash events describing them.

One determinism note for later stages: the page hashes are plain OpenSSL SHA-256 over
token bytes with no per-process seed (`hash_binding.cpp`), so there is no
`PYTHONHASHSEED` analog. Cross-instance key agreement instead depends on the
namespacing inputs, `extra_key`/`cache_salt` on the key (`radix_cache.py:150-235`,
`utils.py:139-183`) and the Mooncake model-aware prefix (PR #31920), which must match
across instances for hits to occur.

## 5. Outcome call: (b), distributes by converging to the content-addressed design

The design decomposes exactly as the research design's central idea anticipated: a
distributed index over a content-addressed store, with the radix layer surviving only
as a per-instance L1/L2 lookup structure. The mechanism, stated precisely:

1. The tree is per-instance by construction; the only shared tier is a key-value store
   of pages (section 2, section 4).
2. Identity at the boundary is a chained SHA-256 page hash, so cross-instance matching
   is a sequential content-addressed prefix probe, the same semantics as vLLM's
   hash-block scheme (section 2).
3. Match granularity at the boundary is `page_size`, and the practical deployment
   value for the Mooncake backend is 64 tokens, coarser than vLLM's 16. Because a
   single `page_size` drives both tiers, the coarsening propagates into the instance
   (section 3).

Outcome (a) is excluded because nothing preserves sub-page matching across the edge.
Outcome (c) is excluded because the tier demonstrably exists and distributes, with a
working Mooncake backend. What is lost at the boundary is precisely the radix
advantage, fine-grained matching, and what survives is what vLLM already had. The
GR-1331 result that granularity dominates transport (LMCache's 256-token chunks at 53%
reuse against 16-token blocks at 98%, `docs/report.md` 6.4) says this 64-token
boundary sits between those two measured points, nearer the healthy end, but the
architectural conclusion stands: **sharing forces the structural cache to become a
content-addressed one, and the token-level advantage stops at, in practice before, the
machine edge.**

Per the research design, Stage 3 does not run. Stage 2 remains worthwhile in narrowed
form: measuring radix (page 1) against page 64 and vLLM's block 16 under partial
prefix overlap would quantify exactly what the boundary costs, which is the number
that makes this outcome concrete.
