# Source document

This project is built over the **my:Optima Secure** health insurance prospectus
(UIN HDFHLIP26058V082526), a publicly available policy document.

The PDF itself is **not committed** to this repository, because it is a third-party copyrighted
document. To run the pipeline, place your own copy here:

```
data/raw/optima_secure_prospectus.pdf
```

Then build the retrieval index:

```
python -m scripts.build_index
```

The prospectus is downloadable from the insurer's public "downloads" section. Any insurer's policy
wording will work with the same pipeline; the clause-numbering and Annexure-A handling in
`src/ingest/` are tuned to this document's structure, so a different document may need adjustments
there.
