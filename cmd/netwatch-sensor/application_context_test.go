package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"netwatch.local/netwatch/internal/contract"
)

func TestApplicationContextJoinsExactServiceWindow(t *testing.T) {
	dir := t.TempDir()
	w := contract.Window{SensorID: "sensor", EntityType: "destination_service", EntityID: "TCP/127.0.0.1:8080",
		WindowStartNS: "5000000000", WindowEndNS: "10000000000", Quality: contract.Quality{Coverage: "complete"}}
	if err := attachApplicationContext(dir, &w); err != nil || w.Application != nil {
		t.Fatalf("missing data: %v", err)
	}
	path := filepath.Join(dir, w.WindowStartNS+".json")
	raw := []byte(`{"schema_version":1,"sensor_id":"sensor","entity_id":"TCP/127.0.0.1:8080","window_start_ns":"5000000000","window_end_ns":"10000000000","completed_requests":0,"server_errors":0,"mean_latency_ms":null,"planned_activity":false}`)
	if err := os.WriteFile(path, raw, 0600); err != nil {
		t.Fatal(err)
	}
	if err := attachApplicationContext(dir, &w); err != nil || w.Application == nil || w.Application.PlannedActivity == nil || *w.Application.PlannedActivity {
		t.Fatalf("verified zero requests: %+v %v", w.Application, err)
	}
	w.EntityID = "TCP/127.0.0.1:9090"
	w.Application = nil
	if err := attachApplicationContext(dir, &w); err == nil || w.Application != nil {
		t.Fatalf("wrong service: %+v %v", w.Application, err)
	}
	w.EntityID = "TCP/127.0.0.1:8080"
	w.Quality.Coverage = "gap"
	if err := attachApplicationContext(dir, &w); err != nil || w.Application == nil {
		t.Fatalf("independent HTTP coverage: %+v %v", w.Application, err)
	}
	w.Application = nil
	w.WindowEndNS = "9000000000"
	if err := attachApplicationContext(dir, &w); err != nil || w.Application != nil {
		t.Fatalf("incomplete time interval: %+v %v", w.Application, err)
	}
}

func TestApplicationOperationsJoinAndKeepUnavailableValues(t *testing.T) {
	dir := t.TempDir()
	w := contract.Window{SensorID: "sensor", EntityType: "destination_service", EntityID: "TCP/127.0.0.1:8080",
		WindowStartNS: "5000000000", WindowEndNS: "10000000000"}
	path := filepath.Join(dir, w.WindowStartNS+".json")
	base := []byte(`{"schema_version":1,"sensor_id":"sensor","entity_id":"TCP/127.0.0.1:8080","window_start_ns":"5000000000","window_end_ns":"10000000000","completed_requests":2,"server_errors":0,"mean_latency_ms":2,"source_ref":"proxy:42","operations":[{"route_id":"reports.export","method":"POST","completed_requests":1,"request_body_bytes":null,"response_body_bytes":42,"expected_requests":0}]}`)
	if err := os.WriteFile(path, base, 0600); err != nil {
		t.Fatal(err)
	}
	if err := attachApplicationContext(dir, &w); err != nil {
		t.Fatal(err)
	}
	if w.Application.SourceRef != "proxy:42" || w.Application.PlannedActivity != nil || len(w.Application.Operations) != 1 {
		t.Fatalf("wrong application context: %+v", w.Application)
	}
	op := w.Application.Operations[0]
	if op.RouteID != "reports.export" || op.Method != "POST" || op.CompletedRequests != 1 || op.RequestBodyBytes != nil ||
		op.ResponseBodyBytes == nil || *op.ResponseBodyBytes != 42 || op.ExpectedRequests == nil || *op.ExpectedRequests != 0 {
		t.Fatalf("wrong operation: %+v", op)
	}

	var invalid map[string]json.RawMessage
	if err := json.Unmarshal(base, &invalid); err != nil {
		t.Fatal(err)
	}
	invalid["operations"] = json.RawMessage(`[{"route_id":"reports.export","method":"POST","completed_requests":1,"request_body_bytes":null,"response_body_bytes":42}]`)
	data, err := json.Marshal(invalid)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	w.Application = nil
	if err := attachApplicationContext(dir, &w); err == nil || w.Application != nil {
		t.Fatalf("accepted incomplete operation: %+v %v", w.Application, err)
	}
}
