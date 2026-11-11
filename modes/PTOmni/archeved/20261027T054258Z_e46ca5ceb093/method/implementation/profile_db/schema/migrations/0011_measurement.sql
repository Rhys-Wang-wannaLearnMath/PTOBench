-- Clean benchmark records retain samples, receipt identity and protocol.
CREATE TABLE IF NOT EXISTS clean_measurement (
    run_id BIGINT PRIMARY KEY,
    document JSON NOT NULL
);
CREATE TABLE IF NOT EXISTS evidence_group (
    group_id VARCHAR PRIMARY KEY,
    kind VARCHAR NOT NULL,
    document JSON NOT NULL
);
