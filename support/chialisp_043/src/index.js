const process = require('process');
const {compile} = require('../pkg/clvm_tools_wasm.js');

if (process.argv.length < 3) {
  console.error('usage: node ./src/index.js (mod ...)');
  process.exit(1);
}

const output = compile(process.argv[2], 'test.clsp', ['.']);
if (output.error) {
  process.stderr.write(output.error);
  process.stderr.write('\n');
  process.exit(1);
}
console.log(output.hex);
