// Package onnxface is the easy, batteries-included face recognition API:
// DownloadModels + Recognizer work in terms of file paths, with no manual
// setup (the onnxruntime shared library and the detection/recognition
// models are located/downloaded automatically). For lower-level control
// (custom detectors/recognizers, image.Image instead of file paths,
// choosing your own comparison metric), see Engine, Compare, and the
// yunet/sface packages -- the same building blocks Recognizer uses
// internally, backed by ONNX Runtime instead of dlib, with models that
// have commercially permissive licenses (YuNet for detection, SFace for
// recognition -- see the README).
package onnxface

import "github.com/leandroveronezi/go-onnxface/face"

/*
InitEnvironment points the package at the onnxruntime shared library and
initializes the ONNX Runtime environment. It's process-global -- call it
once, before using anything else in this package or in yunet/sface.

Most callers don't need this directly: Recognizer.Init calls it
automatically (skipping it if already initialized), using the shared
library DownloadModels fetched. Call InitEnvironment yourself only when
working with the lower-level Engine/yunet/sface directly, or to point at
an onnxruntime install of your own instead of a downloaded one.
*/
func InitEnvironment(sharedLibraryPath string) error {
	return face.InitEnvironment(sharedLibraryPath)
}

/*
CloseEnvironment releases the ONNX Runtime environment. Call it once,
when you're completely done using the package (including any Recognizer
instances) -- it's process-global, not per-instance.
*/
func CloseEnvironment() error {
	return face.CloseEnvironment()
}

// IsInitialized reports whether InitEnvironment has already set up the
// ONNX Runtime environment in this process.
func IsInitialized() bool {
	return face.IsInitialized()
}

// Version returns the onnxruntime version string linked at
// InitEnvironment time.
func Version() string {
	return face.Version()
}
