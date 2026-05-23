# Test fixtures — JUMP Cell Painting brightfield z-stack

Three brightfield images from a real high-content microscopy experiment,
used to test and demonstrate the TIE phase retrieval pipeline on genuine data.

## Source

**Dataset**: JUMP Cell Painting pilot (cpg0000-jump-pilot)  
**S3 bucket**: `cellpainting-gallery` (AWS public, requester-pays)  
**Plate**: `BR00116991`  
**Field of view**: row 01, column 01, field 01 (`r01c01f01`)  
**Instrument**: PerkinElmer Phenix (Andor Zyla sCMOS camera, 20× objective, 2×2 binning)

Original full-resolution files (1080 × 1080, LZW-compressed):

| File | Channel | Name in Harmony |
|---|---|---|
| `r01c01f01p01-ch7sk1fk1fl1.tiff` | ch7 | Brightfield L |
| `r01c01f01p01-ch8sk1fk1fl1.tiff` | ch8 | Brightfield |
| `r01c01f01p01-ch6sk1fk1fl1.tiff` | ch6 | Brightfield H |

Images were centre-cropped to **512 × 512** and saved as uncompressed 16-bit
TIFFs.

## Physical parameters (from Index.idx.xml)

| Parameter | Value |
|---|---|
| Pixel size at sample | 597.98 nm (≈ 598 nm) |
| Illumination wavelength | 740 nm (NIR broadband) |
| z-position ch7 (L) | −4 µm relative to ch8 |
| z-position ch8 (focus) | 0 µm (in-focus) |
| z-position ch6 (H) | +7 µm relative to ch8 |
| Effective dz (half-span) | 5.5 µm |

## Files

| File | TIE role | z-offset |
|---|---|---|
| `brightfield_under.tiff` | I_under | −4 µm (below focus) |
| `brightfield_focus.tiff` | I_focus | 0 µm (in-focus) |
| `brightfield_over.tiff`  | I_over  | +7 µm (above focus) |

## License

JUMP Cell Painting data is released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Please cite the JUMP Cell Painting paper if you use these images.
