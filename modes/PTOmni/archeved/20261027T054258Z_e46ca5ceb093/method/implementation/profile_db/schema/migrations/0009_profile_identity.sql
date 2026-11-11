-- Internal provenance is deliberately separate from agent-facing query facts.
ALTER TABLE run ADD COLUMN identity_key VARCHAR;
CREATE TABLE profile_identity (
    run_id BIGINT PRIMARY KEY,
    document JSON NOT NULL
);
