-- -----------------------------------------------------------------------------
-- PFDB schema migration 0002
--
-- Fix the scheduler-phase shared-queue depth columns: on real level-4
-- captures these are per-queue depth *lists*, not scalars (observed on a
-- Qwen3Decode capture). v1 declared them INTEGER; store the lists as
-- JSON instead, preserving the raw fidelity a false scalar would destroy.
-- -----------------------------------------------------------------------------

ALTER TABLE scheduler_phase DROP COLUMN shared_at_start;
ALTER TABLE scheduler_phase DROP COLUMN shared_at_end;
ALTER TABLE scheduler_phase ADD COLUMN shared_at_start JSON;
ALTER TABLE scheduler_phase ADD COLUMN shared_at_end JSON;