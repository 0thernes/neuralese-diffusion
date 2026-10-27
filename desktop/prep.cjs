// Copies the web app (../site) into www/ for packaging, dropping files that only matter for local dev.
const fs = require('node:fs'); const path = require('node:path');
const src = path.join(__dirname, '..', 'site'), dst = path.join(__dirname, 'www');
fs.rmSync(dst, { recursive: true, force: true }); fs.cpSync(src, dst, { recursive: true, filter: (f) => !/\.b64\.full$/.test(f) });
let n = 0, b = 0; (function walk(d) { for (const e of fs.readdirSync(d, { withFileTypes: true })) { const p = path.join(d, e.name); if (e.isDirectory()) walk(p); else { n++; b += fs.statSync(p).size; } } })(dst);
console.log(`www/: ${n} files, ${(b / 1e6).toFixed(1)} MB`);
