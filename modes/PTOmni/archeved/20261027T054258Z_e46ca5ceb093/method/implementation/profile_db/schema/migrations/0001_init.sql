-- ---------------------------------------------------------------------
-- PFDB schema v1 (migration 0001).
-- One migration creates every table per DESIGN.md 5.2; later milestones
-- fill them (T1 ingest, T3 derived, T8 lifecycle). Primary keys use
-- natural keys where known; uncertain row identities get surrogate ids
-- and may be revised by a later migration.
-- ---------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER PRIMARY KEY,
    applied_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS run (
    run_id          BIGINT PRIMARY KEY,
    program         VARCHAR NOT NULL,
    platform        VARCHAR,
    device_id       INTEGER,
    captured_at     TIMESTAMP,
    swimlane_level  INTEGER,
    clock_freq_hz   BIGINT,
    num_cores       INTEGER,
    core_types      JSON,
    core_to_thread  JSON,
    rank_label      VARCHAR DEFAULT 'single',
    git_commit      VARCHAR,
    git_dirty       BOOLEAN,
    runtime_cfg     JSON,
    cmdline         JSON,
    bench_min_us    DOUBLE,
    bench_median_us DOUBLE,
    bench_mean_us   DOUBLE,
    bench_max_us    DOUBLE,
    bench_rounds    INTEGER,
    makespan_us     DOUBLE,
    raw_span_us     DOUBLE,
    cpm_us          DOUBLE,
    retained        BOOLEAN DEFAULT TRUE,
    notes           VARCHAR,
    tags            VARCHAR[]
);

CREATE TABLE IF NOT EXISTS artifact (
    artifact_id BIGINT PRIMARY KEY,
    run_id      BIGINT NOT NULL,
    kind        VARCHAR NOT NULL,
    rel_path    VARCHAR NOT NULL,
    sha256      VARCHAR,
    size_bytes  BIGINT,
    store_mode  VARCHAR DEFAULT 'link'
);

CREATE TABLE IF NOT EXISTS task (
    run_id               BIGINT NOT NULL,
    task_id              VARCHAR NOT NULL,
    name                 VARCHAR,
    family               VARCHAR,
    engine               VARCHAR,
    scope                VARCHAR,
    early_dispatch_flag  BOOLEAN,
    kernel_ids           JSON,
    block_num            INTEGER,
    num_rows             INTEGER,
    busy_us              DOUBLE,
    wall_us              DOUBLE,
    min_dispatch_us      DOUBLE,
    min_receive_us       DOUBLE,
    min_start_us         DOUBLE,
    max_end_us           DOUBLE,
    max_finish_us        DOUBLE,
    on_cpm_observed      BOOLEAN,
    on_cpm_static        BOOLEAN,
    PRIMARY KEY (run_id, task_id)
);

CREATE TABLE IF NOT EXISTS task_row (
    run_id     BIGINT NOT NULL,
    task_id    VARCHAR NOT NULL,
    core_index INTEGER NOT NULL,
    engine     VARCHAR,
    thread     INTEGER,
    row_index  INTEGER NOT NULL,
    start_us   DOUBLE,
    end_us     DOUBLE,
    aux        INTEGER,
    PRIMARY KEY (run_id, task_id, core_index, row_index)
);

CREATE TABLE IF NOT EXISTS dep_edge (
    edge_id               BIGINT PRIMARY KEY,
    run_id                BIGINT NOT NULL,
    pred                  VARCHAR NOT NULL,
    succ                  VARCHAR NOT NULL,
    source                VARCHAR,
    arg                   VARCHAR,
    flags                 JSON,
    tensor_id             VARCHAR,
    consumer_dtype        VARCHAR,
    consumer_shape        JSON,
    consumer_start_offset VARCHAR,
    consumer_strides      JSON
);

CREATE TABLE IF NOT EXISTS scheduler_phase (
    phase_id        BIGINT PRIMARY KEY,
    run_id          BIGINT NOT NULL,
    lane            INTEGER,
    kind            VARCHAR,
    t0_us           DOUBLE,
    t1_us           DOUBLE,
    loop_iter       INTEGER,
    tasks_processed INTEGER,
    pop_hit         BOOLEAN,
    pop_miss        BOOLEAN,
    shared_at_start INTEGER,
    shared_at_end   INTEGER
);

CREATE TABLE IF NOT EXISTS orch_phase (
    run_id     BIGINT NOT NULL,
    lane       INTEGER NOT NULL,
    submit_idx INTEGER NOT NULL,
    task_id    VARCHAR,
    t0_us      DOUBLE,
    t1_us      DOUBLE,
    PRIMARY KEY (run_id, lane, submit_idx)
);

CREATE TABLE IF NOT EXISTS time_band (
    run_id      BIGINT NOT NULL,
    band_idx    INTEGER NOT NULL,
    t0_us       DOUBLE,
    t1_us       DOUBLE,
    engine      VARCHAR NOT NULL,
    total_cores INTEGER,
    busy_cores  INTEGER,
    task_ids    JSON,
    sparse      BOOLEAN,
    drain_tail  BOOLEAN,
    PRIMARY KEY (run_id, band_idx, engine)
);

CREATE TABLE IF NOT EXISTS idle_gap (
    gap_id         BIGINT PRIMARY KEY,
    run_id         BIGINT NOT NULL,
    engine         VARCHAR,
    core_index     INTEGER,
    t0_us          DOUBLE,
    t1_us          DOUBLE,
    kind           VARCHAR,
    ready_task_ids JSON,
    evidence       VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS cpm_path (
    run_id                BIGINT NOT NULL,
    kind                  VARCHAR NOT NULL,
    seq                   INTEGER NOT NULL,
    task_id               VARCHAR,
    wall_us               DOUBLE,
    busy_us               DOUBLE,
    compute_us            DOUBLE,
    stall_us              DOUBLE,
    gap_us                DOUBLE,
    gap_kind              VARCHAR,
    early_dispatch_proven VARCHAR,
    PRIMARY KEY (run_id, kind, seq)
);

CREATE TABLE IF NOT EXISTS pmu_counter (
    pmu_id       BIGINT PRIMARY KEY,
    run_id       BIGINT NOT NULL,
    task_id      VARCHAR,
    counter      VARCHAR NOT NULL,
    value        DOUBLE,
    total_cycles DOUBLE
);

CREATE TABLE IF NOT EXISTS perf_hint (
    run_id      BIGINT NOT NULL,
    seq         INTEGER NOT NULL,
    text        VARCHAR,
    source_path VARCHAR,
    origin      VARCHAR,
    PRIMARY KEY (run_id, seq)
);

CREATE TABLE IF NOT EXISTS memory_entry (
    memory_id   BIGINT PRIMARY KEY,
    run_id      BIGINT NOT NULL,
    kernel      VARCHAR,
    space       VARCHAR,
    usage       DOUBLE,
    limit_value DOUBLE
);

CREATE TABLE IF NOT EXISTS bench_sample (
    run_id       BIGINT NOT NULL,
    round        INTEGER NOT NULL,
    effective_us DOUBLE,
    PRIMARY KEY (run_id, round)
);

CREATE TABLE IF NOT EXISTS incore_entry (
    run_id     BIGINT NOT NULL,
    kernel     VARCHAR NOT NULL,
    status     VARCHAR,
    export_dir VARCHAR,
    metrics    JSON,
    PRIMARY KEY (run_id, kernel)
);

CREATE TABLE IF NOT EXISTS trial (
    trial_id        BIGINT PRIMARY KEY,
    parent_trial_id BIGINT,
    run_id          BIGINT,
    goal            VARCHAR,
    hypothesis      VARCHAR,
    changed_files   JSON,
    status          VARCHAR,
    verdict         VARCHAR,
    evidence_refs   JSON,
    created_at      TIMESTAMP,
    notes           VARCHAR
);

CREATE TABLE IF NOT EXISTS baseline (
    baseline_id   BIGINT PRIMARY KEY,
    name          VARCHAR,
    program       VARCHAR,
    platform      VARCHAR,
    run_id        BIGINT,
    bench_mean_us DOUBLE,
    criteria      JSON,
    accepted_at   TIMESTAMP
);

-- Non-key access paths (primary keys already index their own columns).
CREATE INDEX IF NOT EXISTS idx_task_family     ON task (family);
CREATE INDEX IF NOT EXISTS idx_task_row_lookup ON task_row (run_id, core_index, start_us);
CREATE INDEX IF NOT EXISTS idx_dep_pred        ON dep_edge (run_id, pred);
CREATE INDEX IF NOT EXISTS idx_dep_succ        ON dep_edge (run_id, succ);
CREATE INDEX IF NOT EXISTS idx_band_lookup     ON time_band (run_id, engine, band_idx);
CREATE INDEX IF NOT EXISTS idx_trial_run       ON trial (run_id);
CREATE INDEX IF NOT EXISTS idx_cpm_lookup      ON cpm_path (run_id, kind, seq);
CREATE INDEX IF NOT EXISTS idx_pmu_run         ON pmu_counter (run_id, task_id);