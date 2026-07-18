package onnxface

import (
	"os"
	"testing"
)

// ortSharedLibraryPath returns the onnxruntime shared library path from the
// ONNXFACE_ORT_LIB environment variable, skipping the test if it's unset --
// this library is a large platform-specific binary we don't check in (see
// the README for how to obtain it), so it isn't present unless a developer
// or CI step has downloaded it.
func ortSharedLibraryPath(t *testing.T) string {
	t.Helper()

	path := os.Getenv("ONNXFACE_ORT_LIB")
	if path == "" {
		t.Skip("ONNXFACE_ORT_LIB not set; see README for how to obtain the onnxruntime shared library")
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("ONNXFACE_ORT_LIB=%s: %v", path, err)
	}

	return path
}

func TestInitClose(t *testing.T) {
	path := ortSharedLibraryPath(t)

	if err := InitEnvironment(path); err != nil {
		t.Fatalf("InitEnvironment: %v", err)
	}
	defer CloseEnvironment()

	v := Version()
	if v == "" {
		t.Errorf("Version() returned an empty string")
	}
	t.Logf("onnxruntime version: %s", v)
}
