package contract

import (
	"encoding/json"
	"os"
	"testing"
)

func TestWindowContract(t *testing.T) {
	b, e := os.ReadFile("../../contracts/examples/window.valid.json")
	if e != nil {
		t.Fatal(e)
	}
	if e = Validate("window", b); e != nil {
		t.Fatal(e)
	}
	cases := []func(map[string]any){
		func(m map[string]any) { m["schema_version"] = 2 },
		func(m map[string]any) { m["window_end_ns"] = m["window_start_ns"] },
		func(m map[string]any) { m["window_start_ns"] = "9999999999999999999" },
		func(m map[string]any) { m["extra"] = "bad" },
		func(m map[string]any) { m["features"].(map[string]any)["out_packets"] = 1.5 },
	}
	for i, change := range cases {
		var m map[string]any
		json.Unmarshal(b, &m)
		change(m)
		raw, _ := json.Marshal(m)
		if Validate("window", raw) == nil {
			t.Fatalf("accepted invalid %d", i)
		}
	}
}
func TestPrediction(t *testing.T) {
	b, _ := os.ReadFile("../../contracts/examples/prediction.valid.json")
	if e := Validate("prediction", b); e != nil {
		t.Fatal(e)
	}
}

func TestDeliveryContracts(t *testing.T) {
	raw, err := os.ReadFile("../../contracts/examples/window.valid.json")
	if err != nil {
		t.Fatal(err)
	}
	var window map[string]any
	json.Unmarshal(raw, &window)
	id := "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	batch := map[string]any{"schema_version": 1, "batch_id": id, "windows": []any{window}, "queue_loss": map[string]any{"count": 0}}
	check := func(kind string, value any, want bool) {
		t.Helper()
		data, _ := json.Marshal(value)
		if err := Validate(kind, data); (err == nil) != want {
			t.Fatalf("%s: %v", kind, err)
		}
	}
	check("batch", batch, true)
	batch["windows"] = []any{window, window}
	check("batch", batch, false)
	batch["windows"] = []any{}
	check("batch", batch, false)
	batch["windows"] = []any{window}
	window["window_end_ns"] = window["window_start_ns"]
	check("batch", batch, false)
	check("ack", map[string]any{"batch_id": id, "accepted_event_ids": []string{"a"}}, true)
	check("ack", map[string]any{"batch_id": id, "accepted_event_ids": []string{"a", "a"}}, false)
	check("rejection", map[string]any{"batch_id": id, "code": "invalid_window", "error_index": 0}, true)
	check("rejection", map[string]any{"batch_id": id, "code": "invalid_window", "error_index": 100}, false)
}

func TestApplicationContextContractRejectsWrongWindowAndInvalidCounts(t *testing.T) {
	raw, err := os.ReadFile("../../contracts/examples/window.valid.json")
	if err != nil {
		t.Fatal(err)
	}
	var w map[string]any
	if err = json.Unmarshal(raw, &w); err != nil {
		t.Fatal(err)
	}
	w["entity_type"], w["entity_id"] = "destination_service", "TCP/127.0.0.1:8080"
	w["window_start_ns"], w["window_end_ns"] = "5000000000", "10000000000"
	app := map[string]any{"completed_requests": 10, "server_errors": 2, "mean_latency_ms": 1.2, "planned_activity": false}
	w["application"] = app
	check := func(want bool) {
		t.Helper()
		data, e := json.Marshal(w)
		if e != nil {
			t.Fatal(e)
		}
		if err := Validate("window", data); (err == nil) != want {
			t.Fatalf("unexpected validation result: %v", err)
		}
	}
	check(true)
	app["server_errors"] = 11
	check(false)
	app["server_errors"] = 2
	w["window_end_ns"] = "9999999999"
	check(false)
}

func TestApplicationOperationContract(t *testing.T) {
	const app = `{"completed_requests":2,"server_errors":0,"mean_latency_ms":1,"planned_activity":false`
	const op = `{"route_id":"reports.export","method":"POST","completed_requests":2,"request_body_bytes":null,"response_body_bytes":20,"expected_requests":0}`
	for name, tc := range map[string]struct {
		fields string
		valid  bool
	}{
		"legacy":       {`}`, true},
		"opaque":       {`,"source_ref":"proxy:42","operations":[` + op + `]}`, true},
		"duplicate":    {`,"operations":[` + op + `,` + op + `]}`, false},
		"too many":     {`,"operations":[` + op + `,{"route_id":"second","method":"GET","completed_requests":1,"request_body_bytes":null,"response_body_bytes":null,"expected_requests":null}]}`, false},
		"bad route":    {`,"operations":[{"route_id":"/reports/123","method":"POST","completed_requests":0,"request_body_bytes":null,"response_body_bytes":null,"expected_requests":2}]}`, false},
		"bad source":   {`,"source_ref":"https://proxy"}`, false},
		"newline":      {`,"source_ref":"proxy\n"}`, false},
		"no body":      {`,"operations":[{"route_id":"reports","method":"POST","completed_requests":0,"request_body_bytes":null,"response_body_bytes":20,"expected_requests":2}]}`, false},
		"expected":     {`,"operations":[{"route_id":"reports","method":"POST","completed_requests":0,"request_body_bytes":0,"response_body_bytes":null,"expected_requests":2}]}`, true},
		"invalid size": {`,"operations":[{"route_id":"reports","method":"POST","completed_requests":1,"request_body_bytes":-1,"response_body_bytes":null,"expected_requests":2}]}`, false},
	} {
		t.Run(name, func(t *testing.T) {
			if err := Validate("application", []byte(app+tc.fields)); (err == nil) != tc.valid {
				t.Fatalf("unexpected application result: %v", err)
			}
		})
	}
}
