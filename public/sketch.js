async function loadTransformers() {
  try {
    const module = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.4.0');
    return module;
  } catch (error) {
    console.error('Failed to load transformers.js', error);
  }
}

async function setup() {
  createCanvas(100, 20);
  background(127);
  let textInput = createInput('cat');

  // Loading Transformers.js
  let tfjs = await loadTransformers();
  let { AutoTokenizer, CLIPTextModelWithProjection, cos_sim } = tfjs;

  // Loading the model
  const tokenizer = await AutoTokenizer.from_pretrained('Xenova/clip-vit-base-patch16');
  const text_model = await CLIPTextModelWithProjection.from_pretrained(
    'Xenova/clip-vit-base-patch16',
    {
      device: 'webgpu',
      dtype: 'fp16',
      progress_callback: (x) => {
        console.log(x);
        if (x.status === 'progress') {
          background(127);
          fill(0);
          noStroke();
          let w = (width * x.progress) / 100;
          // console.log(w);
          rect(0, 0, w, 20);
        }
      },
    }
  );

  // Loading the embeddings from binary file
  // Binary format is much faster to load than JSON for large numerical datasets
  
  // Step 1: Load the raw binary data file
  let rawData = await loadBytes('embeddings/embeddings.bin');
  
  // Step 2: Convert the raw binary data to a Float32Array (32-bit floating point numbers)
  // Each floating point number takes 4 bytes, so we divide the byte length by 4
  // This gives us one long array of all our embedding values
  let rawFloats = new Float32Array(rawData.buffer, rawData.byteOffset, rawData.byteLength / 4);
  
  // Step 3: Set up to split the data back into individual embeddings
  // Each embedding vector has 512 dimensions (values)
  const length = 512;
  
  // Calculate how many individual embeddings we have
  const total = rawFloats.length / length;
  
  // Step 4: Recreate our array of individual embeddings
  let embeddings = [];
  for (let i = 0; i < total; i++) {
    // Calculate where each embedding starts in the long array
    let start = i * length;
    
    // Extract just the 512 values for this embedding
    let embedding = rawFloats.slice(start, start + length);
    
    // Add it to our array of embeddings
    embeddings.push(embedding);
  }

  // Loading photo metadata
  let photos = await loadJSON('embeddings/photo.json');
  console.log(`loaded embeddings for ${photos.length} images`);

  let submit = createButton('submit');
  let imagesDiv = createDiv();
  submit.mousePressed(async () => {
    // clear imagesDiv
    imagesDiv.html('');

    // Prepending 'A photo of a ' helps the model generate accurate embeddings for searching images
    const texts = ['photo of ' + textInput.value().trim()];
    // const texts = [textInput.value().trim()];
    console.log(texts);

    const text_inputs = tokenizer(texts, { padding: true, truncation: true });

    // console.log('Input text:', texts);
    // console.log('Token IDs:', text_inputs.input_ids.tolist());

    // Compute embeddings
    const { text_embeds } = await text_model(text_inputs);
    const text_embeddings = text_embeds.normalize().tolist();

    let similarities = [];
    for (let i = 0; i < embeddings.length; i++) {
      const image_embeddings = embeddings[i];
      const { id, url } = photos[i];
      const similarity = cos_sim(text_embeddings[0], image_embeddings);
      similarities.push({ id, url, similarity });
    }

    similarities.sort((a, b) => b.similarity - a.similarity);
    // console.log(similarities);

    for (let i = 0; i < 50; i++) {
      const { id, url, similarity } = similarities[i];
      // const img = createImg(`images/${photo_id}.jpg`, 'image from unsplash');
      // const img = createImg(`images/${id}.jpg`, 'image from unsplash');
      const img = createImg(url + '?w=512&h=512&fit=max&q=90', 'image from unsplash');
      img.parent(imagesDiv);
      img.size(128, 128);
    }
  });
}
