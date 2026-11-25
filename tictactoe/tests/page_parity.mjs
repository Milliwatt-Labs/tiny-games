// node page_parity.mjs <index.html> <boards.json>
// Runs the page's own <script id="engine"> block (no DOM, no dependencies) with the page's embedded nets and
// prints {"<net>": [move, ...], ...}. boards.json is {"<net>": [[9 ints], ...], ...}: the boards to run each net on
// (every embedded net must have a list).
import { readFileSync } from "node:fs";
import vm from "node:vm";

const [page, boardsPath] = process.argv.slice(2);
const html = readFileSync(page, "utf8");
const pick = (re, what) => {
  const m = html.match(re);
  if (!m) throw new Error(`no ${what} in ${page}`);
  return m[1];
};
const engine = pick(/<script id="engine">([\s\S]*?)<\/script>/, "engine script");
const nets = JSON.parse(pick(/<script type="application\/json" id="nets">([\s\S]*?)<\/script>/, "nets"));
const ctx = vm.createContext({});
vm.runInContext(engine, ctx);
const boards = JSON.parse(readFileSync(boardsPath, "utf8"));
const out = {};
for (const k of Object.keys(nets)) {
  if (!boards[k]) throw new Error(`no boards for net ${k}`);
  out[k] = boards[k].map((b) => ctx.chooseMove(nets[k], b.slice()));
}
process.stdout.write(JSON.stringify(out));
