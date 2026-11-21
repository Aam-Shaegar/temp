package contract

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net"
	"strconv"
	"strings"
	"sync"

	"github.com/santhosh-tekuri/jsonschema/v6"
	"netwatch.local/netwatch/contracts"
)

var validationSchemas = sync.OnceValues(func() (map[string]*jsonschema.Schema, error) {
	kinds := []string{"window", "prediction", "packet", "batch", "ack", "rejection", "inference-request", "inference-response", "security-observation", "security-assessment"}
	compiler := jsonschema.NewCompiler()
	for _, name := range kinds {
		schemaData, err := contracts.Schemas.ReadFile(name + ".schema.json")
		if err != nil {
			return nil, err
		}
		schemaDoc, err := jsonschema.UnmarshalJSON(bytes.NewReader(schemaData))
		if err != nil {
			return nil, err
		}
		if err = compiler.AddResource("https://netwatch.local/"+name+".schema.json", schemaDoc); err != nil {
			return nil, err
		}
	}
	schemas := make(map[string]*jsonschema.Schema, len(kinds))
	for _, kind := range kinds {
		schema, err := compiler.Compile("https://netwatch.local/" + kind + ".schema.json")
		if err != nil {
			return nil, err
		}
		schemas[kind] = schema
	}
	application, err := compiler.Compile("https://netwatch.local/window.schema.json#/properties/application")
	if err != nil {
		return nil, err
	}
	schemas["application"] = application
	return schemas, nil
})

func Validate(kind string, data []byte) error {
	schemas, err := validationSchemas()
	if err != nil {
		return err
	}
	schema := schemas[kind]
	if schema == nil {
		return fmt.Errorf("unknown contract %q", kind)
	}
	doc, err := jsonschema.UnmarshalJSON(bytes.NewReader(data))
	if err != nil {
		return err
	}
	if err = schema.Validate(doc); err != nil {
		return err
	}
	return validateFields(kind, doc.(map[string]any))
}

// Nested objects passed the enclosing schema. Apply their semantic checks
// directly to the parsed values.
func validateFields(kind string, obj map[string]any) error {
	if kind == "application" {
		return validateApplicationFields(obj)
	}
	for _, key := range []string{"window_start_ns", "window_end_ns", "timestamp_ns"} {
		if raw, ok := obj[key]; ok {
			if _, err := strconv.ParseInt(raw.(string), 10, 64); err != nil {
				return fmt.Errorf("%s outside int64", key)
			}
		}
	}
	if kind == "batch" || kind == "inference-request" {
		seen := map[string]bool{}
		for _, value := range obj["windows"].([]any) {
			record := value.(map[string]any)
			if err := validateFields("window", record); err != nil {
				return err
			}
			id := record["event_id"].(string)
			if seen[id] {
				return fmt.Errorf("duplicate batch event_id")
			}
			seen[id] = true
		}
	}
	if kind == "inference-response" {
		seen := map[string]bool{}
		for _, value := range obj["predictions"].([]any) {
			record := value.(map[string]any)
			if err := validateFields("prediction", record); err != nil {
				return err
			}
			id := record["event_id"].(string)
			if seen[id] {
				return fmt.Errorf("duplicate prediction event_id")
			}
			seen[id] = true
		}
	}
	if kind == "packet" && obj["decode_status"] == "decoded" {
		for _, field := range []string{"src_ip", "dst_ip"} {
			if net.ParseIP(obj[field].(string)) == nil {
				return fmt.Errorf("invalid IP")
			}
		}
		if obj["protocol"] == "TCP" || obj["protocol"] == "UDP" {
			if obj["src_port"] == nil || obj["dst_port"] == nil {
				return fmt.Errorf("missing transport port")
			}
		}
	}
	if kind == "window" {
		start, _ := strconv.ParseInt(obj["window_start_ns"].(string), 10, 64)
		end, _ := strconv.ParseInt(obj["window_end_ns"].(string), 10, 64)
		if end <= start {
			return fmt.Errorf("window end must exceed start")
		}
		if app, ok := obj["application"]; ok {
			if obj["entity_type"] != "destination_service" || !strings.HasPrefix(obj["entity_id"].(string), "TCP/") ||
				end-start != 5_000_000_000 || start%5_000_000_000 != 0 {
				return fmt.Errorf("application context requires an aligned TCP service window")
			}
			if err := validateApplicationFields(app.(map[string]any)); err != nil {
				return err
			}
		}
		for name, v := range obj["features"].(map[string]any) {
			if v != nil && (strings.HasSuffix(name, "packets") || strings.HasSuffix(name, "bytes") || strings.HasPrefix(name, "unique_") || strings.HasSuffix(name, "flows") || strings.HasSuffix(name, "attempts") || name == "completed_flow_count") {
				if _, e := v.(json.Number).Int64(); e != nil {
					return fmt.Errorf("%s must be int64", name)
				}
			}
		}
	}
	if kind == "prediction" {
		scored := obj["status"] == "normal" || obj["status"] == "anomalous"
		if scored != (obj["score"] != nil && obj["threshold"] != nil) {
			return fmt.Errorf("score/status mismatch")
		}
		if !scored && (obj["score"] != nil || obj["threshold"] != nil) {
			return fmt.Errorf("unavailable state has score")
		}
	}
	return nil
}

func validateApplicationFields(fields map[string]any) error {
	if latency := fields["mean_latency_ms"]; latency != nil {
		if _, err := latency.(json.Number).Float64(); err != nil {
			return fmt.Errorf("invalid application latency")
		}
	}
	count, countErr := fields["completed_requests"].(json.Number).Int64()
	errors, errorsErr := fields["server_errors"].(json.Number).Int64()
	if countErr != nil || errorsErr != nil || errors > count ||
		(count == 0) != (fields["mean_latency_ms"] == nil) {
		return fmt.Errorf("invalid application counts")
	}
	operations, _ := fields["operations"].([]any)
	seen := make(map[[2]string]struct{}, len(operations))
	remaining := count
	for _, value := range operations {
		item := value.(map[string]any)
		key := [2]string{item["route_id"].(string), item["method"].(string)}
		if _, exists := seen[key]; exists {
			return fmt.Errorf("duplicate application route and method")
		}
		seen[key] = struct{}{}
		completed, err := item["completed_requests"].(json.Number).Int64()
		if err != nil || completed > remaining {
			return fmt.Errorf("application operation counts exceed completed requests")
		}
		remaining -= completed
		for _, field := range []string{"request_body_bytes", "response_body_bytes", "expected_requests"} {
			if number := item[field]; number != nil {
				v, err := number.(json.Number).Int64()
				if err != nil || field != "expected_requests" && completed == 0 && v != 0 {
					return fmt.Errorf("invalid application operation %s", field)
				}
			}
		}
	}
	return nil
}
