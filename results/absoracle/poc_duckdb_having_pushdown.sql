-- Candidate first-report: DuckDB pushes HAVING quals on group keys into
-- the scan without checking saturation over its own grouping-equivalence
-- classes.  Mechanism: src/optimizer/pushdown/pushdown_aggregate.cpp
-- (PushdownAggregate pushes any filter whose bindings are all group
-- columns; no per-class-constancy check).
--
-- Engine: DuckDB 1.5.2 (also 1.1.3).  Not known-fixed; no matching issue
-- (nearest: #6686 grouping-sets NULLs, #20491 EXCEPT rep, #16901 join rep
-- -- all different mechanisms).
--
-- Semantic law: GROUP BY merges +0.0/-0.0 into ONE class (its own `=`
-- agrees).  HAVING evaluates per-group; COUNT(*) must equal class size
-- or the row must be absent.  A count of 1 is impossible.

CREATE TABLE t(f DOUBLE);
INSERT INTO t VALUES (CAST('-0.0' AS DOUBLE)), (0.0);

SELECT f, COUNT(*) FROM t GROUP BY f;
-- (0.0, 2)  -- one group, both members

SELECT f, COUNT(*) FROM t GROUP BY f HAVING CAST(f AS VARCHAR) = '-0.0';
-- expected: (-0.0, 2) or (0.0, 2) if rep passes, else empty
-- actual:   (-0.0, 1)  -- the filter ran pre-grouping, killing one member

SELECT f, COUNT(*) FROM t GROUP BY f HAVING 1.0/f = CAST('-Inf' AS DOUBLE);
-- same violation via a numeric discriminating predicate

-- window form: deterministic (every input row is retained)
SELECT kk, c FROM (
  SELECT f, CAST(f AS VARCHAR) kk, COUNT(*) OVER (PARTITION BY f) c FROM t
) s WHERE CAST(f AS VARCHAR) = '-0.0';
-- expected: (-0.0, 2); actual: (-0.0, 1)

-- EXPLAIN shows the predicate landed on SEQ_SCAN, e.g.
--   Filters: (CAST(f AS VARCHAR) = '-0.0')
-- i.e. evaluated per input row before grouping.
