package face

import (
	ort "github.com/yalue/onnxruntime_go"
)

/*
InitEnvironment points the package at the onnxruntime shared library and
initializes the ONNX Runtime environment. It's process-global -- call it
once, before using anything else in this package, yunet, or sface.

sharedLibraryPath must point at the onnxruntime shared library for your
platform (libonnxruntime.so on Linux, .dylib on macOS, .dll on
Windows). Unlike dlib, this library is not compiled from source: it's a
prebuilt binary published by Microsoft.
*/
func InitEnvironment(sharedLibraryPath string) error {
	ort.SetSharedLibraryPath(sharedLibraryPath)
	return ort.InitializeEnvironment()
}

/*
CloseEnvironment releases the ONNX Runtime environment. Call it once,
when you're completely done using it -- it's process-global, not
per-instance.
*/
func CloseEnvironment() error {
	return ort.DestroyEnvironment()
}

// IsInitialized reports whether InitEnvironment has already set up the
// ONNX Runtime environment in this process.
func IsInitialized() bool {
	return ort.IsInitialized()
}

// Version returns the onnxruntime version string linked at
// InitEnvironment time.
func Version() string {
	return ort.GetVersion()
}
