-- ---------------------------------------------------------------------
-- T9 extended modalities: metadata-only tables for args_dump and
-- scope_stats (DESIGN.md 5.2/8.1). Raw payloads (args.bin, in-core
-- traces, visualize_data.bin) never enter any table or the store; only
-- their parsed metadata is persisted. incore_entry already exists from
-- migration 0001 and is filled by the separate incore ingest.
-- ---------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS args_dump_entry (
    run_id     BIGINT NOT NULL,
    seq        INTEGER NOT NULL,
    task_id    VARCHAR,
    stage      VARCHAR,
    role       VARCHAR,
    arg_index  INTEGER,
    kind       VARCHAR,
    dtype      VARCHAR,
    shape      JSON,
    bin_size   BIGINT,
    PRIMARY KEY (run_id, seq)
);

CREATE TABLE IF NOT EXISTS scope_stats_entry (
    run_id  BIGINT NOT NULL,
    seq     INTEGER NOT NULL,
    site    VARCHAR,
    ring    INTEGER,
    phase   VARCHAR,
    payload JSON,
    PRIMARY KEY (run_id, seq)
);
