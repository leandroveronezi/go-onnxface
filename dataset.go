package onnxface

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/leandroveronezi/go-onnxface/face"
)

// SaveDataset saves Dataset to a JSON file at path.
func (r *Recognizer) SaveDataset(path string) error {

	data, err := func() ([]byte, error) {
		r.mu.RLock()
		defer r.mu.RUnlock()
		return json.Marshal(r.Dataset)
	}()
	if err != nil {
		return err
	}

	return os.WriteFile(path, data, 0600)

}

/*
LoadDataset loads entries from a JSON file saved by SaveDataset, appending
them to the current Dataset -- it doesn't replace it. Identify sees the
loaded entries immediately, with no separate step needed.
*/
func (r *Recognizer) LoadDataset(path string) error {

	if !face.FileExists(path) {
		return fmt.Errorf("%s: file not found", path)
	}

	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	var dataset []Data
	if err := json.NewDecoder(f).Decode(&dataset); err != nil {
		return err
	}

	r.mu.Lock()
	r.Dataset = append(r.Dataset, dataset...)
	r.mu.Unlock()

	return nil

}
