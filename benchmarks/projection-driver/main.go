// The projection driver evaluates the incident state machine on frozen predictions.
package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"

	"netwatch.local/netwatch/internal/incidents"
)

type observation struct {
	incidents.Point
	HTTP *struct{ Status string }
}

type run struct {
	Policy string
	Points []observation
}

type episode struct {
	First       int64  `json:"first"`
	Last        int64  `json:"last"`
	Detected    int64  `json:"detected"`
	End         *int64 `json:"end"`
	Closed      *int64 `json:"closed"`
	Interrupted *int64 `json:"interrupted"`
	Channel     string `json:"evidence_channel"`
}

type stream struct {
	state      incidents.State
	channel    string
	lastSignal int64
}

func (s *stream) update(rows *[]episode, point incidents.Point, policy string) {
	change := s.state.ObserveWithPolicy(point, policy)
	if change.Kind == "opened" {
		var first int64
		s.lastSignal = 0
		for _, evidence := range change.Open {
			if evidence.Signal() {
				if s.lastSignal == 0 {
					first = evidence.Start
				}
				s.lastSignal = evidence.End
			}
		}
		*rows = append(*rows, episode{First: first, Last: point.End, Detected: point.End, Channel: s.channel})
		s.state.Active = int64(len(*rows))
		return
	}
	if change.Kind == "" || s.state.Active == 0 {
		return
	}
	item := &(*rows)[s.state.Active-1]
	if change.Kind == "interrupted" {
		item.Interrupted = change.InterruptedAt
		item.Last = max(item.Last, *change.InterruptedAt)
		s.state.Active = 0
		return
	}
	item.Last = point.End
	if change.Kind == "signal" {
		s.lastSignal = point.End
	}
	if change.Kind == "closed" {
		end, closed := s.lastSignal, point.End
		item.End, item.Closed = &end, &closed
		s.state.Active = 0
	}
}

func project(r run) ([]episode, error) {
	rows := make([]episode, 0)
	switch r.Policy {
	case incidents.LegacyPolicy, incidents.PersistencePolicy, incidents.ConsecutivePolicy, incidents.EvidencePolicy:
	default:
		return nil, errors.New("unsupported projection policy")
	}
	network := stream{channel: "combined"}
	http := stream{channel: "http"}
	if r.Policy == incidents.EvidencePolicy {
		network.channel = "network"
	}
	for _, input := range r.Points {
		p := input.Point
		network.update(&rows, p, r.Policy)
		if r.Policy == incidents.EvidencePolicy {
			var status *string
			if input.HTTP != nil {
				status = &input.HTTP.Status
			}
			http.update(&rows, p.HTTPPoint(status), r.Policy)
		}
	}
	sort.SliceStable(rows, func(i, j int) bool {
		if rows[i].First != rows[j].First {
			return rows[i].First < rows[j].First
		}
		if rows[i].Detected != rows[j].Detected {
			return rows[i].Detected < rows[j].Detected
		}
		return rows[i].Channel < rows[j].Channel
	})
	return rows, nil
}

func main() {
	var input []run
	err := json.NewDecoder(os.Stdin).Decode(&input)
	result := make([][]episode, len(input))
	if err == nil {
		for i, r := range input {
			result[i], err = project(r)
			if err != nil {
				break
			}
		}
	}
	if err == nil {
		err = json.NewEncoder(os.Stdout).Encode(result)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
