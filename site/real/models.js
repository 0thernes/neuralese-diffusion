// Real pretrained tier: every file pinned to a Hugging Face commit and a SHA-256 (computed on the box from the exact bytes; a mismatch aborts).
export const ORT = { version: '1.23.0',
  js: { url: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/ort.webgpu.bundle.min.mjs', sha256: '84681495ff949ab9693b54624b9ba85016fa6a830b46744b47c021d025825416', bytes: 116338 },
  glue: { url: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/ort-wasm-simd-threaded.asyncify.mjs', sha256: 'ceb0d6891b9333176d23387a421e3f19d1ba010ea49332101b734322aa7569f3', bytes: 51913 },
  wasm: { url: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/ort-wasm-simd-threaded.asyncify.wasm', sha256: 'd12253ecce2fb4dd764bfb4d413d23953dfb016f7787f9ea1aaa600404fd1d0b', bytes: 25372571 } };
const hf = (repo, rev, f) => `https://huggingface.co/${repo}/resolve/${rev}/${f}`;
const TOK = { vocab: { url: hf('RanaLLC/tiny-sd-onnx-fp16', '2ceee652906dc5382f069977672b3f2eb41e5ae6', 'tokenizer/vocab.json'), sha256: 'e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349', bytes: 1059962 },
  merges: { url: hf('RanaLLC/tiny-sd-onnx-fp16', '2ceee652906dc5382f069977672b3f2eb41e5ae6', 'tokenizer/merges.txt'), sha256: '9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a', bytes: 524619 } };
export const TAESD = { key: 'taesd', name: 'TAESD decoder (madebyollin, ONNX by julienkay)', license: 'MIT', params: '1.2M', tier: 'phone',
  file: { url: hf('julienkay/taesd', 'c8a437fd0201c21c3bcd298fcf3181b063bcc1eb', 'vae_decoder/model.onnx'), sha256: '0c9459fda7b8d582081be2494adbdaf806366728f49b8ba243f7dc132455ad10', bytes: 4912312 } };
export const PIPES = [
  { key: 'tinysd', name: 'tiny-sd (Segmind, distilled SD-1.5)', repo: 'RanaLLC/tiny-sd-onnx-fp16', license: 'CreativeML OpenRAIL-M', params: '≈0.55B total (UNet 323M fp16)', tier: 'desktop-webgpu',
    cfg: { hidden: 768, spacing: 'leading', pad: 49407, steps: 20, guidance: 7.5 }, tok: TOK,
    files: { text: { url: hf('RanaLLC/tiny-sd-onnx-fp16', '2ceee652906dc5382f069977672b3f2eb41e5ae6', 'text_encoder/model.onnx'), sha256: '10ee490937f4971fb2f34a9c3773153756afe456d5b8e5132918371e60e5b861', bytes: 246368629 },
      unet: { url: hf('RanaLLC/tiny-sd-onnx-fp16', '2ceee652906dc5382f069977672b3f2eb41e5ae6', 'unet/model.onnx'), sha256: '1a6b80f6bc772314ae62dce60f610b2d0ec2abb3f42b8074600228698e46ae44', bytes: 647372306 },
      vae: { url: hf('RanaLLC/tiny-sd-onnx-fp16', '2ceee652906dc5382f069977672b3f2eb41e5ae6', 'vae_decoder/model.onnx'), sha256: '8c041386f53d56f4b8b58e8c116d2b77687e1b533419e754c4c7e44d39fae273', bytes: 99093889 } } },
  { key: 'sdturbo', name: 'SD-Turbo (Stability AI, adversarial-distilled SD-2.1, 1–4 steps)', repo: 'schmuell/sd-turbo-ort-web', license: 'Stability AI Non-Commercial Research Community License', params: '≈1.3B total (UNet 866M fp16)', tier: 'desktop-webgpu',
    cfg: { hidden: 1024, spacing: 'trailing', pad: 0, steps: 1, guidance: 1 }, tok: TOK, ortOpts: { graphOptimizationLevel: 'basic' },
    files: { text: { url: hf('schmuell/sd-turbo-ort-web', 'ace89b7d2cd849f9a73914cdbb8a3ea60c853dd1', 'text_encoder/model.onnx'), sha256: '5d1f5bd3286abf26313a9e87db6fd32e47d8b1b2a84ffdf470884c3fe89497c4', bytes: 681393168 },
      unet: { url: hf('schmuell/sd-turbo-ort-web', 'ace89b7d2cd849f9a73914cdbb8a3ea60c853dd1', 'unet/model.onnx'), sha256: '7468133d9f7babec9cf6636e160188fe25aad206792c0abdaa722edc676963c4', bytes: 1733430199 },
      vae: { url: hf('schmuell/sd-turbo-ort-web', 'ace89b7d2cd849f9a73914cdbb8a3ea60c853dd1', 'vae_decoder/model.onnx'), sha256: 'b37a85f91a982811957abd6cfb617499742309f7b8e253240da3132e533a7605', bytes: 99094314 } } },
];
export const totalBytes = (p) => Object.values(p.files).reduce((s, f) => s + f.bytes, 0) + TAESD.file.bytes + p.tok.vocab.bytes + p.tok.merges.bytes;
