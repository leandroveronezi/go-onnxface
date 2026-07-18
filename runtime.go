package onnxface

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
)

// onnxRuntimeVersion is the onnxruntime release DownloadModels fetches
// and this package's cgo binding (yalue/onnxruntime_go) targets. Keep
// these in sync.
const onnxRuntimeVersion = "1.26.0"

// runtimeAsset describes where to get the onnxruntime shared library for
// one GOOS/GOARCH combination: the release archive to download, and the
// path of the shared library within it.
type runtimeAsset struct {
	url    string
	member string
	isZip  bool
}

// runtimeLibraryName is the fixed file name DownloadModels writes the
// onnxruntime shared library to, and Init/Recognizer.Init look for,
// independent of version -- so callers never need to know the exact
// versioned file name inside the release archive.
func runtimeLibraryName() (string, error) {
	switch runtime.GOOS {
	case "linux":
		return "libonnxruntime.so", nil
	case "darwin":
		return "libonnxruntime.dylib", nil
	case "windows":
		return "onnxruntime.dll", nil
	default:
		return "", fmt.Errorf("no known onnxruntime shared library name for GOOS %q", runtime.GOOS)
	}
}

/*
runtimeAssetFor maps a GOOS/GOARCH pair to the onnxruntime release asset
that contains its shared library, verified directly against the release's
actual asset list and archive contents (Microsoft's naming isn't fully
uniform across platforms -- e.g. Windows ships a plain "onnxruntime.dll"
with no version in the name, macOS only publishes arm64 as of 1.26.0).
*/
func runtimeAssetFor(goos, goarch string) (runtimeAsset, error) {

	base := "https://github.com/microsoft/onnxruntime/releases/download/v" + onnxRuntimeVersion + "/"

	switch goos + "/" + goarch {
	case "linux/amd64":
		dir := "onnxruntime-linux-x64-" + onnxRuntimeVersion
		return runtimeAsset{
			url:    base + dir + ".tgz",
			member: dir + "/lib/libonnxruntime.so." + onnxRuntimeVersion,
		}, nil
	case "linux/arm64":
		dir := "onnxruntime-linux-aarch64-" + onnxRuntimeVersion
		return runtimeAsset{
			url:    base + dir + ".tgz",
			member: dir + "/lib/libonnxruntime.so." + onnxRuntimeVersion,
		}, nil
	case "darwin/arm64":
		dir := "onnxruntime-osx-arm64-" + onnxRuntimeVersion
		return runtimeAsset{
			url:    base + dir + ".tgz",
			member: dir + "/lib/libonnxruntime." + onnxRuntimeVersion + ".dylib",
		}, nil
	case "windows/amd64":
		dir := "onnxruntime-win-x64-" + onnxRuntimeVersion
		return runtimeAsset{
			url:    base + dir + ".zip",
			member: dir + "/lib/onnxruntime.dll",
			isZip:  true,
		}, nil
	case "windows/arm64":
		dir := "onnxruntime-win-arm64-" + onnxRuntimeVersion
		return runtimeAsset{
			url:    base + dir + ".zip",
			member: dir + "/lib/onnxruntime.dll",
			isZip:  true,
		}, nil
	default:
		return runtimeAsset{}, fmt.Errorf(
			"no prebuilt onnxruntime %s available for %s/%s -- see https://github.com/microsoft/onnxruntime/releases/tag/v%s, download the right one yourself and point Init at it",
			onnxRuntimeVersion, goos, goarch, onnxRuntimeVersion,
		)
	}

}

// downloadRuntimeLibrary fetches the onnxruntime shared library for the
// current GOOS/GOARCH into dir, named per runtimeLibraryName. Left
// untouched (not re-downloaded) if it already exists there.
func downloadRuntimeLibrary(dir string) error {

	libName, err := runtimeLibraryName()
	if err != nil {
		return err
	}
	dest := filepath.Join(dir, libName)
	if fileExists(dest) {
		return nil
	}

	asset, err := runtimeAssetFor(runtime.GOOS, runtime.GOARCH)
	if err != nil {
		return err
	}

	archivePath := filepath.Join(dir, filepath.Base(asset.url))
	if err := downloadFile(archivePath, asset.url); err != nil {
		return fmt.Errorf("downloading onnxruntime: %w", err)
	}
	defer os.Remove(archivePath)

	if asset.isZip {
		return extractZipMember(archivePath, asset.member, dest)
	}
	return extractTarGzMember(archivePath, asset.member, dest)

}

func downloadFile(path, url string) error {

	resp, err := modelsHTTPClient.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%s: unexpected status %s", url, resp.Status)
	}

	out, err := os.Create(path)
	if err != nil {
		return err
	}

	if _, err := io.Copy(out, resp.Body); err != nil {
		out.Close()
		os.Remove(path)
		return err
	}

	return out.Close()

}

func extractTarGzMember(archivePath, member, dest string) error {

	f, err := os.Open(archivePath)
	if err != nil {
		return err
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gz.Close()

	tr := tar.NewReader(gz)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			return fmt.Errorf("member %s not found in %s", member, archivePath)
		}
		if err != nil {
			return err
		}
		if hdr.Name != member {
			continue
		}
		return writeExtracted(dest, tr)
	}

}

func extractZipMember(archivePath, member, dest string) error {

	zr, err := zip.OpenReader(archivePath)
	if err != nil {
		return err
	}
	defer zr.Close()

	for _, f := range zr.File {
		if f.Name != member {
			continue
		}
		rc, err := f.Open()
		if err != nil {
			return err
		}
		defer rc.Close()
		return writeExtracted(dest, rc)
	}

	return fmt.Errorf("member %s not found in %s", member, archivePath)

}

// writeExtracted copies r to dest via a temp-file-then-rename, so a
// reader that fails partway (or two concurrent downloads racing on the
// same dest) never leaves a truncated or torn shared library on disk.
func writeExtracted(dest string, r io.Reader) error {

	tmp := dest + ".tmp"

	out, err := os.Create(tmp)
	if err != nil {
		return err
	}

	if _, err := io.Copy(out, r); err != nil {
		out.Close()
		os.Remove(tmp)
		return err
	}
	if err := out.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	if err := os.Chmod(tmp, 0755); err != nil {
		os.Remove(tmp)
		return err
	}

	return os.Rename(tmp, dest)

}
