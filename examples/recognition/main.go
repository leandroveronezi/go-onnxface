// Command recognition demonstrates the easy, batteries-included API:
// onnxface.Recognizer, working entirely in terms of image file paths.
package main

import (
	"errors"
	"fmt"
	"path/filepath"

	"github.com/leandroveronezi/go-onnxface"
)

const fotosDir = "../fotos"
const modelsDir = "../../models"

func addFile(rec *onnxface.Recognizer, path, id string) {
	err := rec.AddImageToDataset(path, id)
	switch {
	case errors.Is(err, onnxface.ErrNoFace), errors.Is(err, onnxface.ErrMultipleFaces):
		fmt.Printf("%s: not exactly one face, skipping\n", path)
	case err != nil:
		fmt.Println(err)
	}
}

func main() {

	rec := &onnxface.Recognizer{}

	// Safe to call on every run: fetches the onnxruntime shared library
	// and the YuNet/SFace models into modelsDir only if they're not
	// already there.
	if err := rec.DownloadModels(modelsDir); err != nil {
		fmt.Println(err)
		return
	}
	if err := rec.Init(modelsDir); err != nil {
		fmt.Println(err)
		return
	}
	defer rec.Close()

	addFile(rec, filepath.Join(fotosDir, "amy.jpg"), "Amy")
	addFile(rec, filepath.Join(fotosDir, "bernadette.jpg"), "Bernadette")
	addFile(rec, filepath.Join(fotosDir, "howard.jpg"), "Howard")
	addFile(rec, filepath.Join(fotosDir, "penny.jpg"), "Penny")
	addFile(rec, filepath.Join(fotosDir, "raj.jpg"), "Raj")
	addFile(rec, filepath.Join(fotosDir, "sheldon.jpg"), "Sheldon")
	addFile(rec, filepath.Join(fotosDir, "leonard.jpg"), "Leonard")

	results, err := rec.IdentifyMultiples(filepath.Join(fotosDir, "elenco3.jpg"))
	if err != nil {
		fmt.Println(err)
		return
	}

	for _, r := range results {
		fmt.Printf("%s: distance=%.4f confidence=%.2f%%\n", r.Id, r.Distance, r.Confidence*100)
	}

}
