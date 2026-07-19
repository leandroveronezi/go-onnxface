package face

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

var downloadHTTPClient = &http.Client{Timeout: 5 * time.Minute}

// FileExists reports whether path exists and is a regular file (not a
// directory).
func FileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

// DownloadFile fetches url and writes it to path. path is left absent
// (not a partial file) if anything fails partway. Shared by every
// package's DownloadModel -- each engine downloads only its own model,
// not a bundle of everything, so a caller that only uses one engine
// never pays for the others.
func DownloadFile(path, url string) error {

	resp, err := downloadHTTPClient.Get(url)
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
