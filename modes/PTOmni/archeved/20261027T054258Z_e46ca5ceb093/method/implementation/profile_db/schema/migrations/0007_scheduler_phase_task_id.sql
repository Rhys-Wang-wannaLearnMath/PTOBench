-- ---------------------------------------------------------------------
-- Scheduler schema (simpler 6940c8cc): scheduler_records.streams[].records
-- carry the acted-on task token on dummy_task/predicated_skip and
-- graph_prepare kinds. The column holds the normalized canonical id so
-- dummy/early-dispatch analysis can join phases back to tasks; phases of
-- every other kind (and archived captures) stay NULL.
--
-- The same change also made the dispatch pop counters counts: the host
-- emits how many queue pops hit/missed inside the record's window (e.g.
-- pop_hit=2, pop_miss=11), not a boolean. The BOOLEAN columns silently
-- truncated those to true, so they become INTEGER; archived flat-schema
-- captures stored true/false and read back as 1/0.
-- ---------------------------------------------------------------------

ALTER TABLE scheduler_phase ADD COLUMN task_id VARCHAR;
ALTER TABLE scheduler_phase ALTER COLUMN pop_hit TYPE INTEGER USING CASE WHEN pop_hit THEN 1 WHEN pop_hit IS NULL THEN NULL ELSE 0 END;
ALTER TABLE scheduler_phase ALTER COLUMN pop_miss TYPE INTEGER USING CASE WHEN pop_miss THEN 1 WHEN pop_miss IS NULL THEN NULL ELSE 0 END;
