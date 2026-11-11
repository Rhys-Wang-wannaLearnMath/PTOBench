-- ---------------------------------------------------------------------
-- T3 derived layer: per-row AICPU dispatch/receive/finish timestamps.
-- Level-2+ captures carry these on every physical row (the AICore<->AICPU
-- join delivers them); level-1 rows keep the converter-synthesized 0.0
-- values already used for the task-level aggregates. The columns feed
-- the early-dispatch proof (per-block dispatch vs observed ready) and the
-- earliest-row stall decomposition.
-- ---------------------------------------------------------------------

ALTER TABLE task_row ADD COLUMN dispatch_us DOUBLE;
ALTER TABLE task_row ADD COLUMN receive_us DOUBLE;
ALTER TABLE task_row ADD COLUMN finish_us DOUBLE;