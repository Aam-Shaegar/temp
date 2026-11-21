package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/netip"
	"os"
	"path/filepath"
	"testing"

	"netwatch.local/netwatch/internal/contract"
	"netwatch.local/netwatch/internal/window"
)

func TestContextOnlyExportDoesNotInventPacketMeasurements(t *testing.T) {
	dir := t.TempDir()
	raw := []byte(`{"schema_version":1,"sensor_id":"sensor","entity_id":"TCP/127.0.0.1:8080","window_start_ns":"5000000000","window_end_ns":"10000000000","completed_requests":10,"server_errors":8,"mean_latency_ms":17,"planned_activity":null}`)
	if err := os.WriteFile(filepath.Join(dir, "5000000000.json"), raw, 0600); err != nil {
		t.Fatal(err)
	}
	engine, err := window.New(window.Config{Service: netip.MustParseAddrPort("127.0.0.1:8080"), Protocol: "TCP",
		FeatureSet: contract.TrafficV2, MaxUnique: 1, MaxPending: 1})
	if err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	if err = exportContextWindows(context.Background(), &output, engine, dir, "sensor", "proxy-only", 0, 15_000_000_000); err != nil {
		t.Fatal(err)
	}
	data := bytes.TrimSpace(output.Bytes())
	if err = contract.Validate("window", data); err != nil {
		t.Fatal(err)
	}
	var got contract.Window
	if err = json.Unmarshal(data, &got); err != nil || got.Quality.Coverage != "gap" || got.Quality.CaptureDropped != nil ||
		got.Application == nil || got.Application.ServerErrors != 8 || got.CaptureEpoch != "proxy-only" {
		t.Fatal(string(data), err)
	}
	for name, value := range got.Features {
		if value != nil {
			t.Fatalf("fabricated measurement %s", name)
		}
	}
}
