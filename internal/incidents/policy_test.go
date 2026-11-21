package incidents

import "testing"

func TestLifecycle(t *testing.T) {
	var s State
	tick := int64(0)
	step := func(status string, want string) {
		t.Helper()
		score, threshold := 0.5, 1.
		if status == "anomalous" {
			score = 1.2
		}
		if status == "near" {
			score = .97
			status = "normal"
		}
		c := s.Observe(Point{ID: "window", Start: tick, End: tick + 5, Status: status, Score: &score, Threshold: &threshold})
		tick += 5
		if c.Kind != want {
			t.Fatalf("%s: got %s, want %s", status, c.Kind, want)
		}
		if c.Kind == "opened" {
			if len(c.Open) != 3 {
				t.Fatal("missing evidence")
			}
			s.Active = 1
		}
	}
	step("anomalous", "")
	tick += 5
	step("normal", "")
	step("anomalous", "")
	step("anomalous", "opened")
	step("anomalous", "signal")
	step("normal", "observed")
	step("insufficient_data", "unknown")
	step("normal", "observed")
	step("near", "observed")
	step("normal", "observed")
	step("normal", "observed")
	step("normal", "closed")
	if c := s.Observe(Point{Start: 1, End: 6}); c.Kind != "" || s.Late != 1 {
		t.Fatal("late window changed history")
	}
}

func TestPersistencePolicyClosesWithoutAnIsolatedSpike(t *testing.T) {
	var s State
	threshold := 1.
	for i, score := range []float64{1.2, 1.3, 1.1, .99, .98, 1.05} {
		status := "normal"
		if score > threshold {
			status = "anomalous"
		}
		start := int64(i * 5)
		change := s.ObserveWithPolicy(Point{Start: start, End: start + 5, Status: status,
			Score: &score, Threshold: &threshold}, PersistencePolicy)
		switch i {
		case 2:
			if change.Kind != "opened" {
				t.Fatalf("third window did not open the incident: %+v", change)
			}
			s.Active = 1
		case 5:
			if change.Kind != "closed" {
				t.Fatalf("isolated spike prevented recovery confirmation: %+v", change)
			}
		default:
			if change.Kind == "signal" {
				t.Fatalf("unexpected extension at window %d", i)
			}
		}
	}
}

func TestPersistencePolicySeparatesObservationGapsFromRecovery(t *testing.T) {
	threshold, score := 1., 1.2
	for _, point := range []Point{
		{Start: 15, End: 20, Status: "projection_gap"},
		{Start: 25, End: 30, Status: "anomalous", Score: &score, Threshold: &threshold},
	} {
		s := State{End: 15, Active: 1, Recoveries: 2}
		change := s.ObserveWithPolicy(point, PersistencePolicy)
		if change.Kind != "interrupted" || change.InterruptedAt == nil || *change.InterruptedAt != 15 ||
			change.LinkWindow != (point.Status == "projection_gap") || s.Recoveries != 0 {
			t.Fatalf("unexpected gap transition: %+v, state %+v", change, s)
		}
	}
}

func TestConsecutivePolicySeparatesNearbyEpisodes(t *testing.T) {
	var state State
	for i, want := range []string{"", "", "", "opened", "observed", "closed", "", "opened", "interrupted", "", "opened"} {
		status, score, threshold := "anomalous", 1.2, 1.
		switch i {
		case 1, 4, 5:
			status, score = "normal", .9
		case 8:
			status = "insufficient_data"
		}
		start := int64(i * 5)
		point := Point{ID: want, Start: start, End: start + 5, Status: status, Score: &score, Threshold: &threshold}
		change := state.ObserveWithPolicy(point, ConsecutivePolicy)
		if change.Kind != want {
			t.Fatalf("window %d: got %+v, want %q", i, change, want)
		}
		switch want {
		case "opened":
			if len(change.Open) != 2 || change.Open[0].Start != start-5 {
				t.Fatalf("window %d: stale evidence: %+v", i, change.Open)
			}
			state.Active = int64(i)
		case "closed", "interrupted":
			state.Active = 0
		}
	}
}
