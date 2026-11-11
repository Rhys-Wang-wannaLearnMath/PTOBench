-- ---------------------------------------------------------------------
-- Trial bench-only evidence: store unprofiled PYPTO_BENCH numbers on a
-- trial that has no ingested run (compile_error / bench-only campaigns).
-- ---------------------------------------------------------------------

ALTER TABLE trial ADD COLUMN IF NOT EXISTS bench_min_us DOUBLE;
ALTER TABLE trial ADD COLUMN IF NOT EXISTS bench_median_us DOUBLE;
ALTER TABLE trial ADD COLUMN IF NOT EXISTS bench_mean_us DOUBLE;
ALTER TABLE trial ADD COLUMN IF NOT EXISTS bench_max_us DOUBLE;
ALTER TABLE trial ADD COLUMN IF NOT EXISTS bench_rounds INTEGER;
