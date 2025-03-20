import fs from 'fs';
import csv from 'csv-parser';
import axios from 'axios';
import path from 'path';
import { fileURLToPath } from 'url';
import { AutoProcessor, CLIPVisionModelWithProjection, RawImage } from '@huggingface/transformers';

// Get directory path
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function readTSV(filePath) {
  const results = [];
  return new Promise((resolve, reject) => {
    fs.createReadStream(filePath)
      .pipe(csv({ separator: '\t' }))
      .on('data', (data) => results.push(data))
      .on('end', () => resolve(results))
      .on('error', (err) => reject(err));
  });
}

// Helper function to download images asynchronously
async function downloadImage(url, filename) {
  const imagesDir = path.resolve(__dirname, 'public/images');

  // Ensure the images directory exists
  if (!fs.existsSync(imagesDir)) {
    fs.mkdirSync(imagesDir);
  }

  const filePath = path.join(imagesDir, filename);
  const writer = fs.createWriteStream(filePath);

  const response = await axios.get(url, { responseType: 'stream' });
  response.data.pipe(writer);

  return new Promise((resolve, reject) => {
    writer.on('finish', () => resolve(filePath));
    writer.on('error', reject);
  });
}

// if partially downloaded try
// rm -rf node_modules/@huggingface/transformers/.cache

const processor = await AutoProcessor.from_pretrained('Xenova/clip-vit-base-patch16');
const vision_model = await CLIPVisionModelWithProjection.from_pretrained(
  'Xenova/clip-vit-base-patch16',
  {
    // device: 'webgpu',
    dtype: 'fp32',
    progress_callback: (x) => process.stdout.write(x.status + ' ' + x.progress + '\r'),
  }
);

const photosTSV = await readTSV('unsplash/photos.tsv000');
console.log(`Processing ${photosTSV.length} photos from unsplash`);

let photos = [];
let embeddings = [];

for (let i = 0; i < photosTSV.length; i++) {
  const photo = photosTSV[i];

  // Adding URL parameter ?w=512&q=80 to resize and compress the image
  const imageUrl = `${photo.photo_image_url}?w=1024&h=1024&fit=max&q=90`;
  const filename = `${photo.photo_id}.jpg`;
  const filePath = path.resolve(__dirname, 'public/images', filename);

  // Only download the image if it doesn't exist
  // No need to actually down the images but good to have the option
  // if (!fs.existsSync(filePath)) {
  //   log(`Downloading ${filename}: ${i + 1}/${photos.length}\r`);
  //   await downloadImage(imageUrl, filename);
  // }

  // Process the embedding
  try {
    const image = await RawImage.read(imageUrl);
    const image_inputs = await processor(image);
    const { image_embeds } = await vision_model(image_inputs);
    const image_embeddings = image_embeds.normalize().tolist();

    // Add to arrays
    embeddings.push(image_embeddings[0]);
    photos.push({ id: photo.photo_id, url: photo.photo_image_url });

    log(`Computed embeddings: ${i + 1}/${photosTSV.length}\r`);
  } catch (e) {
    console.error(e);
  }

  // Save every 500 cycles since it takes a long time to run
  if (i % 500 == 0) {
    writeData();
  }
}

// If we get to the end
writeData();

function writeData() {
  let flattened = embeddings.flat();
  const embeddingsBuffer = Buffer.from(new Float32Array(flattened).buffer);
  fs.writeFileSync('public/embeddings/embeddings.bin', embeddingsBuffer);
  fs.writeFileSync('public/embeddings/photo.json', JSON.stringify(photos));
  console.log(`Saved ${embeddings.length} embeddings and photo IDs.`);
}

// Helper function to log progress
function log(message) {
  process.stdout.clearLine();
  process.stdout.cursorTo(0);
  process.stdout.write(message);
}
