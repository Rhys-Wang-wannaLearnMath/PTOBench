-- ---------------------------------------------------------------------
-- PFDB dogfooding follow-up: normalized task ids, PMU sample provenance,
-- optional-modality states, and benchmark invocation strata.
-- ---------------------------------------------------------------------

ALTER TABLE task ADD COLUMN IF NOT EXISTS task_id_raw VARCHAR;
ALTER TABLE task ADD COLUMN IF NOT EXISTS task_id_u64 VARCHAR;
ALTER TABLE task_row ADD COLUMN IF NOT EXISTS task_id_raw VARCHAR;
ALTER TABLE task_row ADD COLUMN IF NOT EXISTS task_id_u64 VARCHAR;
ALTER TABLE dep_edge ADD COLUMN IF NOT EXISTS pred_u64 VARCHAR;
ALTER TABLE dep_edge ADD COLUMN IF NOT EXISTS succ_u64 VARCHAR;
ALTER TABLE dep_edge ADD COLUMN IF NOT EXISTS pred_raw VARCHAR;
ALTER TABLE dep_edge ADD COLUMN IF NOT EXISTS succ_raw VARCHAR;

ALTER TABLE pmu_counter ADD COLUMN IF NOT EXISTS task_id_raw VARCHAR;
ALTER TABLE pmu_counter ADD COLUMN IF NOT EXISTS task_id_u64 VARCHAR;
ALTER TABLE pmu_counter ADD COLUMN IF NOT EXISTS sample_seq INTEGER;
ALTER TABLE pmu_counter ADD COLUMN IF NOT EXISTS thread_id INTEGER;
ALTER TABLE pmu_counter ADD COLUMN IF NOT EXISTS core_id INTEGER;
ALTER TABLE pmu_counter ADD COLUMN IF NOT EXISTS func_id INTEGER;
ALTER TABLE pmu_counter ADD COLUMN IF NOT EXISTS core_type VARCHAR;
ALTER TABLE pmu_counter ADD COLUMN IF NOT EXISTS event_type VARCHAR;
ALTER TABLE args_dump_entry ADD COLUMN IF NOT EXISTS task_id_raw VARCHAR;
ALTER TABLE args_dump_entry ADD COLUMN IF NOT EXISTS task_id_u64 VARCHAR;

UPDATE task
SET task_id_raw = COALESCE(task_id_raw, task_id),
    task_id_u64 = COALESCE(task_id_u64, CAST(try_cast(task_id AS UBIGINT) AS VARCHAR)),
    task_id = CASE
        WHEN try_cast(task_id AS UBIGINT) IS NOT NULL THEN CAST(try_cast(task_id AS UBIGINT) AS VARCHAR)
        ELSE task_id
    END;

UPDATE task_row
SET task_id_raw = COALESCE(task_id_raw, task_id),
    task_id_u64 = COALESCE(task_id_u64, CAST(try_cast(task_id AS UBIGINT) AS VARCHAR)),
    task_id = CASE
        WHEN try_cast(task_id AS UBIGINT) IS NOT NULL THEN CAST(try_cast(task_id AS UBIGINT) AS VARCHAR)
        ELSE task_id
    END;

UPDATE dep_edge
SET pred_raw = COALESCE(pred_raw, pred),
    succ_raw = COALESCE(succ_raw, succ),
    pred_u64 = COALESCE(pred_u64, CAST(try_cast(pred AS UBIGINT) AS VARCHAR)),
    succ_u64 = COALESCE(succ_u64, CAST(try_cast(succ AS UBIGINT) AS VARCHAR)),
    pred = CASE WHEN try_cast(pred AS UBIGINT) IS NOT NULL THEN CAST(try_cast(pred AS UBIGINT) AS VARCHAR) ELSE pred END,
    succ = CASE WHEN try_cast(succ AS UBIGINT) IS NOT NULL THEN CAST(try_cast(succ AS UBIGINT) AS VARCHAR) ELSE succ END;

UPDATE pmu_counter
SET task_id_raw = COALESCE(task_id_raw, task_id),
    task_id_u64 = COALESCE(task_id_u64, CAST(try_cast(task_id AS UBIGINT) AS VARCHAR)),
    task_id = CASE
        WHEN try_cast(task_id AS UBIGINT) IS NOT NULL THEN CAST(try_cast(task_id AS UBIGINT) AS VARCHAR)
        ELSE task_id
    END,
    -- v4 rows had no sample coordinate; all counters from that legacy
    -- aggregate are one logical sample rather than one sample per counter.
    sample_seq = COALESCE(sample_seq, 0),
    event_type = COALESCE(event_type, 'legacy');

UPDATE args_dump_entry
SET task_id_raw = COALESCE(task_id_raw, task_id),
    task_id_u64 = COALESCE(task_id_u64, CAST(try_cast(task_id AS UBIGINT) AS VARCHAR)),
    task_id = CASE
        WHEN try_cast(task_id AS UBIGINT) IS NOT NULL THEN CAST(try_cast(task_id AS UBIGINT) AS VARCHAR)
        ELSE task_id
    END;

CREATE TABLE IF NOT EXISTS modality_status (
    run_id        BIGINT NOT NULL,
    modality      VARCHAR NOT NULL,
    requested     BOOLEAN,
    request_value VARCHAR,
    rel_path      VARCHAR,
    size_bytes    BIGINT,
    parser_state  VARCHAR NOT NULL,
    entry_count   BIGINT,
    state         VARCHAR NOT NULL,
    reason        VARCHAR,
    PRIMARY KEY (run_id, modality)
);

CREATE TABLE IF NOT EXISTS bench_stratum (
    run_id           BIGINT NOT NULL,
    stratum          INTEGER NOT NULL,
    source_sha256    VARCHAR,
    rounds           INTEGER,
    warmup           INTEGER,
    rank_count       INTEGER,
    aggregation_mode VARCHAR,
    PRIMARY KEY (run_id, stratum)
);

ALTER TABLE bench_sample RENAME TO bench_sample_v4;
CREATE TABLE bench_sample (
    run_id       BIGINT NOT NULL,
    stratum      INTEGER NOT NULL DEFAULT 0,
    round        INTEGER NOT NULL,
    effective_us DOUBLE,
    PRIMARY KEY (run_id, stratum, round)
);
INSERT INTO bench_sample (run_id, stratum, round, effective_us)
SELECT run_id, 0, round, effective_us FROM bench_sample_v4;
DROP TABLE bench_sample_v4;

CREATE INDEX IF NOT EXISTS idx_task_u64 ON task (run_id, task_id_u64);
CREATE INDEX IF NOT EXISTS idx_pmu_task_u64 ON pmu_counter (run_id, task_id_u64);
CREATE INDEX IF NOT EXISTS idx_args_task_u64 ON args_dump_entry (run_id, task_id_u64);
CREATE INDEX IF NOT EXISTS idx_modality_run ON modality_status (run_id, modality);
