# OBS ONNX Runtime Staging Package

This directory is a clean GitHub upload staging package for the OBS ONNX plugin work.

## Included

- Plugin source and project files
- Project documentation and scripts
- Staging manifest (`STAGING_MANIFEST.txt`)

## Not Included

To keep this repository lightweight and license-safe, the following are intentionally excluded:

- OBS source tree (`obs-studio`)
- Prebuilt ONNX Runtime binaries
- DirectML/other large vendor runtime packages
- Large dependency bundles and local build artifacts

## How To Get OBS Source

Clone OBS separately:

```powershell
git clone https://github.com/obsproject/obs-studio.git
cd obs-studio
git submodule update --init --recursive
```

Then follow OBS build instructions from the official repository (`BUILD.md`).

## How To Get Runtime Libraries

Download required runtime binaries from official sources:

- ONNX Runtime releases: https://github.com/microsoft/onnxruntime/releases
- Microsoft DirectML/runtime components: official Microsoft distribution channels

Recommended workflow:

1. Keep vendor/runtime binaries outside this source repo.
2. Record exact versions and URLs in release notes.
3. If needed, publish binaries as GitHub Release assets instead of committing them to source.

## Notes

- Use `STAGING_MANIFEST.txt` to review staged contents before upload.
- This staging folder is intended for source upload, not as a full binary distribution bundle.
