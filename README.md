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

## What is CLIP?

CLIP (Contrastive Language-Image Pre-training) is a neural network model developed by OpenAI that connects text and images. CLIP was trained on 400 million image-text pairs from a dataset called [WebImageText (WIT)](https://github.com/google-research-datasets/wit).

- [OpenAI CLIP Paper](https://arxiv.org/abs/2103.00020)
- [OpenAI CLIP Blog Post](https://openai.com/research/clip)
- [Hugging Face CLIP Documentation](https://huggingface.co/docs/transformers/model_doc/clip)
- 🎥 [How AI 'Understands' Images (CLIP) - Computerphile](https://youtu.be/KcSXcpluDe4)

## Saving and Loading Binary Data

This project stores embeddings as binary data (`.bin` files) instead of JSON or text. Binary data is more efficient for large numerical datasets like embeddings.

### Saving Embeddings

1. Flatten our nested array of embeddings into one long array of numbers
2. Convert it to a typed `Float32Array` (binary format for 32-bit floating point numbers)
3. Create a Buffer from this binary data
4. Write the buffer to a file

```javascript
// Take our array of arrays and flatten it into one long array
let flattened = embeddings.flat();

// Convert to a binary Float32Array (32-bit floating point numbers)
// This is much more efficient than storing as text/JSON
const embeddingsBuffer = Buffer.from(new Float32Array(flattened).buffer);

// Write the binary data to a file
fs.writeFileSync('public/embeddings/embeddings.bin', embeddingsBuffer);
```

### Loading Embeddings

When loading in p5.js:

1. Load the raw binary data with `loadBytes()`
2. Convert it back to a typed `Float32Array`
3. Split it back into individual embeddings of 512 values each

```javascript
// Load the binary file
let rawData = await loadBytes('embeddings/embeddings.bin');

// Convert the raw bytes back to floating point numbers
// We divide by 4 because each float is 4 bytes
let rawFloats = new Float32Array(rawData.buffer, rawData.byteOffset, rawData.byteLength / 4);

// Each embedding is 512 values long
const length = 512;

// Calculate how many embeddings we have in total
const total = rawFloats.length / length;

// Recreate our array of embeddings
let embeddings = [];
for (let i = 0; i < total; i++) {
  // Extract each 512-value chunk
  let start = i * length;
  let embedding = rawFloats.slice(start, start + length);
  embeddings.push(embedding);
}
```
