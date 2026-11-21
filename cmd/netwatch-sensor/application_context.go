package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"netwatch.local/netwatch/internal/contract"
)

const maxApplicationContextSize = 16 * 1024

// Join at capture time, before writing the immutable spool record. A missing
// aggregate is different from an observed interval with zero requests.
func attachApplicationContext(dir string, w *contract.Window) error {
	start, startErr := strconv.ParseInt(w.WindowStartNS, 10, 64)
	end, endErr := strconv.ParseInt(w.WindowEndNS, 10, 64)
	if dir == "" || w.EntityType != "destination_service" || !strings.HasPrefix(w.EntityID, "TCP/") ||
		startErr != nil || endErr != nil || start < 0 || end <= start || start%5000000000 != 0 || end-start != 5000000000 {
		return nil
	}
	f, err := os.Open(filepath.Join(dir, w.WindowStartNS+".json"))
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil || !info.Mode().IsRegular() || info.Size() > maxApplicationContextSize {
		return fmt.Errorf("invalid context file")
	}
	raw, err := io.ReadAll(io.LimitReader(f, maxApplicationContextSize+1))
	if err != nil || len(raw) > maxApplicationContextSize {
		return fmt.Errorf("invalid context file")
	}
	var entry map[string]json.RawMessage
	if err := json.Unmarshal(raw, &entry); err != nil {
		return fmt.Errorf("invalid application context: %w", err)
	}
	var version int
	if err := json.Unmarshal(entry["schema_version"], &version); err != nil || version != 1 {
		return fmt.Errorf("unsupported application context version")
	}
	delete(entry, "schema_version")
	for _, field := range []struct{ name, expected string }{
		{"sensor_id", w.SensorID}, {"entity_id", w.EntityID},
		{"window_start_ns", w.WindowStartNS}, {"window_end_ns", w.WindowEndNS},
	} {
		var value string
		if err := json.Unmarshal(entry[field.name], &value); err != nil || value == "" || value != field.expected {
			return fmt.Errorf("application context %s does not match the window", field.name)
		}
		delete(entry, field.name)
	}
	// Legacy exporters could omit these null values in their flat files.
	for _, name := range []string{"mean_latency_ms", "planned_activity"} {
		if _, present := entry[name]; !present {
			entry[name] = json.RawMessage("null")
		}
	}
	// Validate the remaining fields with the same schema as window.application.
	aggregate, err := json.Marshal(entry)
	if err != nil {
		return fmt.Errorf("invalid application context: %w", err)
	}
	if err := contract.Validate("application", aggregate); err != nil {
		return fmt.Errorf("invalid application context: %w", err)
	}
	var app contract.ApplicationContext
	if err := json.Unmarshal(aggregate, &app); err != nil {
		return fmt.Errorf("invalid application context: %w", err)
	}
	w.Application = &app
	return nil
}
