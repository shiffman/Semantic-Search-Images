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
    dtype: 'q8',
    progress_callback: (x) => process.stdout.write(x.status + ' ' + x.progress + '\r'),
  }
);

const photos = await readTSV('unsplash/photos.tsv000');
console.log(`Loaded ${photos.length} photos`);

let dbase = {};

// check if dbase file exists already
if (fs.existsSync('public/embeddings/dbase.json')) {
  console.log('Loading existing dbase.json');
  dbase = JSON.parse(fs.readFileSync('public/embeddings/dbase.json'));
  console.log(`Loaded ${Object.keys(dbase).length} embeddings`);
}

// for (let i = 0; i < photos.length; i++) {
for (let i = 0; i < 500; i++) {
  const photo = photos[i];
  // add URL parameter ?w=512&q=80 to resize and compress the image
  const imageUrl = `${photo.photo_image_url}?w=1024&h=1024&fit=max&q=90`;
  const filename = `${photo.photo_id}.jpg`;

  const filePath = path.resolve(__dirname, 'public/images', filename);
  if (dbase[photo.photo_id] && fs.existsSync(filePath)) {
    process.stdout.write(`Skipping ${filename}\r`);
    continue;
  }

  process.stdout.write(`Downloading ${filename}: ${i + 1}/${photos.length}\r`);
  await downloadImage(imageUrl, filename);

  const image = await RawImage.read(imageUrl);
  const image_inputs = await processor(image);

  const { image_embeds } = await vision_model(image_inputs);
  const image_embeddings = image_embeds.normalize().tolist();
  dbase[photo.photo_id] = image_embeddings[0];
}
console.log('');
fs.writeFileSync('public/embeddings/dbase.json', JSON.stringify(dbase));
console.log(`Saved embeddings to dbase.json`);
