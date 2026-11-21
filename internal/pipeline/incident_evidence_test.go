package pipeline_test

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"testing"
	"time"

	"netwatch.local/netwatch/internal/contract"
	"netwatch.local/netwatch/internal/store"
)

func TestIndependentEvidenceStoresConcurrentIncidentsAndKeepsHTTPAliveAcrossCaptureLoss(t *testing.T) {
	db, _ := database(t)
	ctx := context.Background()
	base := time.Unix(1700000000, 0)
	var input []json.RawMessage
	for i := range 5 {
		var item contract.Window
		if err := json.Unmarshal(window(t, fmt.Sprint(i), base.Add(time.Duration(i+1)*5*time.Second)), &item); err != nil {
			t.Fatal(err)
		}
		item.EntityType, item.EntityID = "destination_service", "TCP/127.0.0.1:8080"
		latency, responseBytes := 12., int64(41)
		item.Application = &contract.ApplicationContext{CompletedRequests: 2, MeanLatencyMS: &latency,
			SourceRef: "proxy:42", Operations: []contract.HTTPOperation{{RouteID: "reports.export", Method: "POST", CompletedRequests: 1, ResponseBodyBytes: &responseBytes}}}
		raw, err := json.Marshal(item)
		if err != nil {
			t.Fatal(err)
		}
		if err := contract.Validate("window", raw); err != nil {
			t.Fatal(err)
		}
		input = append(input, raw)
	}
	if err := db.Ingest(ctx, batch(t, input...), sensor, "test-model", "v4"); err != nil {
		t.Fatal(err)
	}
	for range input {
		j, err := db.Claim(ctx, time.Minute, false)
		if err != nil || j == nil {
			t.Fatal(j, err)
		}
		p := prediction(*j)
		p.HTTP = &contract.HTTPEvidence{Status: "normal", Reason: "within_limits"}
		if j.EventID < "3" {
			p.HTTP = &contract.HTTPEvidence{Status: "anomalous", Reason: "latency"}
		}
		if j.EventID < "2" {
			high := .9
			p.Status, p.Score = "anomalous", &high
		}
		if j.EventID == "2" {
			p.Status, p.UnavailableReason, p.Score, p.Threshold = "insufficient_data", "capture_loss", nil, nil
		}
		if err = db.Finish(ctx, *j, p); err != nil {
			t.Fatal(err)
		}
	}
	advanced, err := db.Project(ctx)
	if err != nil || !advanced {
		t.Fatal(advanced, err)
	}
	list, err := db.IncidentList(ctx, store.IncidentFilter{})
	if err != nil || len(list) != 2 {
		t.Fatalf("list=%s err=%v", list, err)
	}
	expectedConfirmation := fmt.Sprint(base.Add(10 * time.Second).UnixNano())
	expectedEnd := fmt.Sprint(base.Add(15 * time.Second).UnixNano())
	expectedInterrupted := fmt.Sprint(base.Add(10 * time.Second).UnixNano())
	for _, item := range list {
		var incident struct {
			ID          string  `json:"id"`
			Channel     string  `json:"evidence_channel"`
			Detected    string  `json:"detected_ns"`
			End         *string `json:"estimated_end_ns"`
			Interrupted *string `json:"interrupted_ns"`
		}
		if err := json.Unmarshal(item, &incident); err != nil || incident.Detected != expectedConfirmation {
			t.Fatal(string(item), err)
		}
		switch incident.Channel {
		case "http":
			if incident.End == nil || *incident.End != expectedEnd || incident.Interrupted != nil {
				t.Fatal(string(item))
			}
		case "network":
			if incident.Interrupted == nil || *incident.Interrupted != expectedInterrupted || incident.End != nil {
				t.Fatal(string(item))
			}
		default:
			t.Fatal(string(item))
		}
		id, err := strconv.ParseInt(incident.ID, 10, 64)
		if err != nil {
			t.Fatal(err)
		}
		journal, err := db.Incident(ctx, id, -1)
		if err != nil {
			t.Fatal(err)
		}
		events := journal["events"].([]json.RawMessage)
		if len(events) == 0 {
			t.Fatal("incident has no evidence windows")
		}
		var evidence struct {
			Application *contract.ApplicationContext `json:"application"`
		}
		if err := json.Unmarshal(events[0], &evidence); err != nil || evidence.Application == nil ||
			evidence.Application.SourceRef != "proxy:42" || len(evidence.Application.Operations) != 1 {
			t.Fatalf("missing proxy evidence: %s %v", events[0], err)
		}
		op := evidence.Application.Operations[0]
		if op.RouteID != "reports.export" || op.RequestBodyBytes != nil || op.ResponseBodyBytes == nil || *op.ResponseBodyBytes != 41 {
			t.Fatalf("unexpected operation evidence: %+v", op)
		}
	}
}
