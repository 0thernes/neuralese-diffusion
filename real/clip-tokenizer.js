// CLIP BPE tokenizer (byte-level BPE with </w> word endings), written from the OpenAI CLIP reference. No dependencies.
// Parity with Hugging Face CLIPTokenizer is checked in test/v2.test.mjs on a fixed prompt list.
function bytesToUnicode() { const bs = [], cs = []; for (let i = 33; i <= 126; i++) bs.push(i); for (let i = 161; i <= 172; i++) bs.push(i); for (let i = 174; i <= 255; i++) bs.push(i);
  cs.push(...bs); let n = 0; for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n++); } const m = {}; bs.forEach((b, i) => (m[b] = String.fromCharCode(cs[i]))); return m; }
const PAT = /<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+/giu;
export class ClipTokenizer {
  constructor(vocab, mergesTxt, { pad = 49407 } = {}) { this.enc = vocab; this.ranks = new Map(); mergesTxt.split('\n').slice(1).filter((l) => l.trim()).slice(0, 49152 - 256 - 2).forEach((l, i) => this.ranks.set(l, i));
    this.b2u = bytesToUnicode(); this.cache = new Map(); this.bos = 49406; this.eos = 49407; this.pad = pad; this.dec = Object.fromEntries(Object.entries(vocab).map(([k, v]) => [v, k])); }
  bpe(tok) { if (this.cache.has(tok)) return this.cache.get(tok); let w = [...tok.slice(0, -1), tok.slice(-1) + '</w>'];
    if (w.length === 1) return [tok + '</w>'];
    for (;;) { let best = null, br = Infinity; for (let i = 0; i < w.length - 1; i++) { const r = this.ranks.get(w[i] + ' ' + w[i + 1]); if (r !== undefined && r < br) { br = r; best = i; } }
      if (best === null) break; const a = w[best], b = w[best + 1], nw = []; for (let i = 0; i < w.length; ) { if (i < w.length - 1 && w[i] === a && w[i + 1] === b) { nw.push(a + b); i += 2; } else nw.push(w[i++]); } w = nw; if (w.length === 1) break; }
    this.cache.set(tok, w); return w; }
  encodeText(text) { const t = text.replace(/\s+/g, ' ').trim().toLowerCase(), ids = [];
    for (const m of t.matchAll(PAT)) { const u = [...new TextEncoder().encode(m[0])].map((b) => this.b2u[b]).join(''); for (const p of this.bpe(u)) ids.push(this.enc[p]); } return ids; }
  // [bos, ...ids, eos, pad...] length 77; also returns the per-position token strings (for attribution labels)
  encode(text, len = 77) { const ids = this.encodeText(text).slice(0, len - 2), full = [this.bos, ...ids, this.eos]; const n = full.length; while (full.length < len) full.push(this.pad);
    return { ids: full, n, tokens: full.slice(0, n).map((i) => (this.dec[i] || '?').replace('</w>', '')) }; }
}
