# Semantic Search for Images

This project will run with any image dataset, the embeddings included are for 25,000 images from the [Unsplash image dataset](https://unsplash.com/data). To generate the embeddings, first install node dependencies.

```bash
npm install
```

Then download the [Unsplash dataset files](https://github.com/unsplash/datasets) and place them in the `unsplash` directory.

```bash
node process-data.js
```

This will:

- Process images from the Unsplash dataset
- Generate CLIP embeddings for each image (mode: [clip-vit-base-patch16](https://huggingface.co/Xenova/clip-vit-base-patch16))
- Save embeddings to `public/embeddings/embeddings.bin`
- Save image metadata to `public/embeddings/photo.json`

To run the p5.js sketch, start a local server to serve the `public` directory
