ALTER TABLE incidents ADD COLUMN evidence_channel text NOT NULL DEFAULT 'combined'
 CHECK (evidence_channel IN ('combined','network','http'));
ALTER TABLE incidents DROP CONSTRAINT incidents_stream_id_detected_ns_key;
ALTER TABLE incidents ADD UNIQUE(stream_id,detected_ns,evidence_channel);
