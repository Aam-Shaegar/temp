-- Removing the channel would discard evidence or collide with a concurrent incident.
DO $$
BEGIN
 IF EXISTS (SELECT 1 FROM incidents WHERE evidence_channel <> 'combined')
    OR EXISTS (SELECT 1 FROM incident_streams WHERE policy_version = 'v4') THEN
  RAISE EXCEPTION 'v4 journal data exists; archive it before reversing this migration';
 END IF;
END $$;
ALTER TABLE incidents DROP CONSTRAINT incidents_stream_id_detected_ns_evidence_channel_key;
ALTER TABLE incidents DROP COLUMN evidence_channel;
ALTER TABLE incidents ADD UNIQUE(stream_id,detected_ns);
