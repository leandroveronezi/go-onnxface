// Package onnxface provides face detection and recognition backed by
// ONNX Runtime, using models with commercially permissive licenses
// (YuNet for detection, SFace for recognition -- see the README).
package onnxface

import (
	ort "github.com/yalue/onnxruntime_go"
)

/*
Init points the package at the onnxruntime shared library and
initializes the ONNX Runtime environment. Call it once, before using
anything else in this package.

sharedLibraryPath must point at the onnxruntime shared library for your
platform (libonnxruntime.so on Linux, .dylib on macOS, .dll on
Windows) -- see the README for how to obtain it. Unlike dlib, this
library is not compiled from source: it's a prebuilt binary published
by Microsoft.
*/
func Init(sharedLibraryPath string) error {
	ort.SetSharedLibraryPath(sharedLibraryPath)
	return ort.InitializeEnvironment()
}

/*
Close releases the ONNX Runtime environment. Call it once, when you're
done using the package. Don't use the package after calling Close.
*/
func Close() error {
	return ort.DestroyEnvironment()
}

// Version returns the onnxruntime version string linked at Init time.
func Version() string {
	return ort.GetVersion()
}
