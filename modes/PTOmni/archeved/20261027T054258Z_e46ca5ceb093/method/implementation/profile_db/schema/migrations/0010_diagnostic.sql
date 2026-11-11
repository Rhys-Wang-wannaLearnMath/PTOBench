-- PMU/compiler captures do not require invented swimlane records.
CREATE TABLE IF NOT EXISTS diagnostic_metric (
    run_id BIGINT NOT NULL,
    metric VARCHAR NOT NULL,
    value DOUBLE NOT NULL,
    unit VARCHAR NOT NULL,
    PRIMARY KEY (run_id, metric)
);
CREATE TABLE IF NOT EXISTS diagnostic_protocol (
    run_id BIGINT PRIMARY KEY,
    mode VARCHAR NOT NULL,
    parser_version VARCHAR NOT NULL,
    signature VARCHAR NOT NULL
);
CREATE TABLE IF NOT EXISTS repair_audit (
    run_id BIGINT NOT NULL,
    repaired_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    document JSON NOT NULL
);

-- Refresh cannot certify physical rows written by older parsers.
ALTER TABLE run ADD COLUMN capture_parser_version VARCHAR;
