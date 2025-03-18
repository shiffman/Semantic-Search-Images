async function loadTransformers() {
  try {
    const module = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.4.0');
    return module;
  } catch (error) {
    console.error('Failed to load transformers.js', error);
  }
}

async function setup() {
  noCanvas();
  let tfjs = await loadTransformers();
  let { AutoTokenizer, CLIPTextModelWithProjection, cos_sim } = tfjs;

  console.log('load text model');
  const tokenizer = await AutoTokenizer.from_pretrained('Xenova/clip-vit-base-patch16');
  const text_model = await CLIPTextModelWithProjection.from_pretrained(
    'Xenova/clip-vit-base-patch16',
    {
      device: 'webgpu',
      dtype: 'q8',
      progress_callback: (x) => console.log(x.status + ' ' + x.progress),
    }
  );

  // Run tokenization
  const texts = ['cat'];
  const text_inputs = tokenizer(texts, { padding: true, truncation: true });

  // Compute embeddings
  const { text_embeds } = await text_model(text_inputs);
  const text_embeddings = text_embeds.normalize().tolist();
  console.log(text_embeddings);

  let dbase = await loadJSON('embeddings/dbase.json');
  let keys = Object.keys(dbase);

  let similarities = [];
  for (let i = 0; i < keys.length; i++) {
    const key = keys[i];
    const image_embeddings = dbase[key];
    const similarity = cos_sim(text_embeddings[0], image_embeddings);
    similarities.push({ key, similarity });
  }

  similarities.sort((a, b) => b.similarity - a.similarity);
  // console.log(similarities);

  for (let i = 0; i < 10; i++) {
    const { key, similarity } = similarities[i];
    const image = createImg(`images/${key}.jpg`, 'image from unsplash');
    image.size(128, 128);
    console.log(key, similarity);
  }
}
