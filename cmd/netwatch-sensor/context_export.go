package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/netip"
	"os"

	"netwatch.local/netwatch/internal/contract"
	"netwatch.local/netwatch/internal/window"
)

const contextWindowNS = int64(5_000_000_000)

// contextWindows exports measured proxy intervals during a packet sensor outage.
// Sender credentials and normal window validation still apply at ingest.
func contextWindows(ctx context.Context, args []string) error {
	opts := flag.NewFlagSet("context", flag.ContinueOnError)
	dir := opts.String("input", "", "read-only directory with verified five-second proxy aggregates")
	sensor := opts.String("sensor", "", "sensor identity used by the aggregates")
	epoch := opts.String("epoch", "", "dedicated context-only observation epoch")
	addr := opts.String("service", "", "TCP destination as IP:port")
	version := opts.String("feature-set", contract.TrafficV2, "feature contract used by the configured detector")
	start := opts.Int64("start-ns", -1, "first aligned event time (inclusive)")
	end := opts.Int64("end-ns", -1, "last aligned event time (exclusive)")
	if err := opts.Parse(args); err != nil {
		return err
	}
	service, err := netip.ParseAddrPort(*addr)
	if opts.NArg() != 0 || *dir == "" || *sensor == "" || *epoch == "" || err != nil || *start < 0 ||
		*end <= *start || *start%contextWindowNS != 0 || *end%contextWindowNS != 0 {
		return fmt.Errorf("context requires input, sensor, an independent epoch, a TCP service and an aligned time range")
	}
	if info, err := os.Stat(*dir); err != nil || !info.IsDir() {
		return fmt.Errorf("context directory unavailable")
	}
	engine, err := window.New(window.Config{Service: service, Protocol: "TCP", FeatureSet: *version, MaxUnique: 1, MaxPending: 1})
	if err != nil {
		return err
	}
	return exportContextWindows(ctx, os.Stdout, engine, *dir, *sensor, *epoch, *start, *end)
}

func exportContextWindows(ctx context.Context, output io.Writer, engine *window.Engine, dir, sensor, epoch string, start, end int64) error {
	enc := json.NewEncoder(output)
	for stamp := start; stamp < end; stamp += contextWindowNS {
		if err := ctx.Err(); err != nil {
			return err
		}
		w, err := engine.Export(window.Result{Start: stamp, End: stamp + contextWindowNS, Coverage: "gap"}, sensor, epoch)
		if err != nil {
			return err
		}
		if err := attachApplicationContext(dir, &w); err != nil {
			return err
		}
		if w.Application == nil {
			continue
		}
		for key := range w.Features {
			w.Features[key] = nil
		}
		if err := enc.Encode(w); err != nil {
			return err
		}
	}
	return nil
}
